use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use reqwest::Client;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Chat;
use rig::providers::openai;
use sqlx::PgPool;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::calibration::CalibrationCurve;
use crate::config::AppConfig;
use crate::data::models::{GammaMarket, PriceTick};
use crate::pricing::kelly::fractional_kelly;
use crate::storage::portfolio::{BetContext, BetSide};

use super::news::{NewsAggregator, NewsItem, NewsMatch};

const GAMMA_API: &str = "https://gamma-api.polymarket.com";
const CLOB_API: &str = "https://clob.polymarket.com";

#[derive(Debug, Clone)]
pub struct Signal {
    pub market_id: String,
    pub question: String,
    pub side: BetSide,
    pub current_price: f64,
    pub estimated_prob: f64,
    pub confidence: f64,
    pub edge: f64,
    pub kelly_size: f64,
    pub reasoning: String,
    pub end_date: Option<String>,
    pub volume: f64,
    pub context: BetContext,
}

impl Signal {
    pub fn to_telegram_message(&self) -> String {
        let (emoji, side_label) = match self.side {
            BetSide::Yes => ("\u{1f7e2}", "YES"),
            BetSide::No => ("\u{1f534}", "NO"),
        };

        // Format multi-agent reasoning as separate lines
        let reasoning_lines: Vec<&str> = self.reasoning.split(" | ").collect();
        let reasoning_block = if reasoning_lines.len() > 1 {
            let mut block = String::from("\n\u{1f9e0} *Agent Consensus:*\n");
            for line in &reasoning_lines {
                let safe = sanitize_markdown(line);
                block.push_str(&format!("  \u{2022} _{safe}_\n"));
            }
            block
        } else {
            format!("\n\u{1f4a1} _{}_", sanitize_markdown(&self.reasoning))
        };

        format!(
            "{emoji} *{side_label} Signal*\n\n\
             \u{1f4cb} *{question}*\n\n\
             \u{1f4b0} Current price: `{price:.1}\u{00a2}`\n\
             \u{1f3af} Our estimate: `{est:.1}%`\n\
             \u{1f4ca} Edge: `+{edge:.1}%`\n\
             \u{1f512} Confidence: `{conf:.0}%`\n\
             \u{1f4d0} Kelly size: `{kelly:.1}%` of bankroll\n\
             \u{1f4a7} Volume: `${vol:.0}`\n\
             \u{23f0} Expires: {end}\
             {reasoning}",
            question = self.question,
            price = self.current_price * 100.0,
            est = self.estimated_prob * 100.0,
            edge = self.edge * 100.0,
            conf = self.confidence * 100.0,
            kelly = self.kelly_size * 100.0,
            vol = self.volume,
            end = self.end_date.as_deref().unwrap_or("N/A"),
            reasoning = reasoning_block,
        )
    }

    pub fn score(&self) -> f64 {
        self.edge * self.confidence * self.kelly_size
    }
}

pub struct ScanResult {
    pub signals: Vec<Signal>,
    pub rejections: Vec<RejectedSignal>,
    pub markets_scanned: usize,
    pub news_total: usize,
    pub news_new: usize,
}

/// A signal that was evaluated by the LLM but didn't pass the scanner gate.
#[derive(Debug, Clone)]
pub struct RejectedSignal {
    pub question: String,
    pub reason: String,
}

/// Agent roles for multi-agent consensus.
#[derive(Debug, Clone, Copy)]
pub enum AgentRole {
    /// Conservative: assumes market is efficient, looks for reasons news is priced in.
    Skeptic,
    /// Aggressive: looks for catalysts the market hasn't absorbed yet.
    Catalyst,
    /// Statistical: ignores narratives, focuses on base rates and historical patterns.
    BaseRate,
}

impl AgentRole {
    fn label(self) -> &'static str {
        match self {
            Self::Skeptic => "skeptic",
            Self::Catalyst => "catalyst",
            Self::BaseRate => "base_rate",
        }
    }

    fn temperature(self) -> f64 {
        match self {
            Self::Skeptic => 0.1,
            Self::Catalyst => 0.3,
            Self::BaseRate => 0.2,
        }
    }

    fn preamble(self) -> &'static str {
        match self {
            Self::Skeptic => {
                "You are a SKEPTICAL prediction market analyst.\n\n\
                 Your default assumption: the market price is already correct. News is usually \
                 already priced in. Your job is to find reasons why the current price is RIGHT \
                 and push back against overreaction to headlines.\n\n\
                 Only give high confidence (>0.7) when you find genuinely new, market-moving \
                 information that clearly hasn't been absorbed.\n\n\
                 Respond with ONLY a JSON object:\n\
                 {\"probability\": 0.XX, \"confidence\": 0.XX, \"reasoning\": \"one sentence\"}\n\n\
                 - probability: true probability of YES (0.01-0.99)\n\
                 - confidence: 0.0 (no useful info) to 1.0 (news clearly changes outcome)\n\
                 - reasoning: explain why the market IS or ISN'T already priced correctly"
            }
            Self::Catalyst => {
                "You are an AGGRESSIVE prediction market analyst looking for catalysts.\n\n\
                 Your job: find fresh information that the market hasn't absorbed yet. \
                 Look for breaking news, policy changes, data releases, and events that \
                 clearly shift probabilities.\n\n\
                 You can profit from BOTH sides:\n\
                 - If news makes YES more likely than the current price suggests -> buy YES\n\
                 - If news makes YES less likely than the current price suggests -> buy NO\n\n\
                 Be calibrated, but don't be afraid to express conviction when the evidence is strong.\n\n\
                 Respond with ONLY a JSON object:\n\
                 {\"probability\": 0.XX, \"confidence\": 0.XX, \"reasoning\": \"one sentence\"}\n\n\
                 - probability: true probability of YES (0.01-0.99)\n\
                 - confidence: 0.0 (no useful info) to 1.0 (news clearly changes outcome)\n\
                 - reasoning: explain what catalyst matters and why price should move"
            }
            Self::BaseRate => {
                "You are a STATISTICAL prediction market analyst.\n\n\
                 Ignore narratives and hype. Focus on:\n\
                 1. Historical base rates: how often do events like this happen?\n\
                 2. Price history momentum: is the trend meaningful or noise?\n\
                 3. Time to expiry: how much can the probability realistically move?\n\n\
                 Anchor to base rates first, then adjust for news only if it provides \
                 strong statistical evidence.\n\n\
                 Respond with ONLY a JSON object:\n\
                 {\"probability\": 0.XX, \"confidence\": 0.XX, \"reasoning\": \"one sentence\"}\n\n\
                 - probability: true probability of YES (0.01-0.99)\n\
                 - confidence: 0.0 (no useful info) to 1.0 (news clearly changes outcome)\n\
                 - reasoning: explain the base rate and any statistical adjustment"
            }
        }
    }

    /// Return the first N roles to use for consensus.
    fn roles_for(n: usize) -> Vec<Self> {
        let all = [Self::Skeptic, Self::Catalyst, Self::BaseRate];
        all.into_iter().take(n.clamp(1, 3)).collect()
    }
}

pub struct LiveScanner {
    http: Client,
    openai_client: openai::Client,
    news: NewsAggregator,
    pool: PgPool,
    calibration: RwLock<CalibrationCurve>,
    cfg: Arc<AppConfig>,
}

impl LiveScanner {
    pub async fn new(cfg: &Arc<AppConfig>, pool: PgPool) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client");
        let calibration = CalibrationCurve::load(&pool, cfg.calibration_min_samples).await?;
        Ok(Self {
            news: NewsAggregator::new(http.clone()),
            http,
            openai_client: openai::Client::from_env(),
            pool,
            calibration: RwLock::new(calibration),
            cfg: Arc::clone(cfg),
        })
    }

    /// Reload calibration curve from DB (call periodically).
    pub async fn reload_calibration(&self) -> Result<()> {
        let curve = CalibrationCurve::load(&self.pool, self.cfg.calibration_min_samples).await?;
        *self.calibration.write().await = curve;
        Ok(())
    }

    /// Build an LLM agent for a specific role.
    fn build_agent(
        &self,
        role: AgentRole,
    ) -> rig::agent::Agent<openai::responses_api::ResponsesCompletionModel> {
        self.openai_client
            .agent(&self.cfg.llm_model)
            .preamble(role.preamble())
            .temperature(role.temperature())
            .build()
    }

    pub async fn check_market_resolution(&self, market_id: &str) -> Result<Option<bool>> {
        let url = format!("{GAMMA_API}/markets/{market_id}");
        let resp = self.http.get(&url).send().await?;
        let text = resp.text().await?;
        let market: GammaMarket = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse market {market_id}"))?;
        Ok(market.resolved_yes())
    }

    async fn fetch_book_depth(&self, token_id: &str) -> Result<f64> {
        let url = format!("{CLOB_API}/book?token_id={token_id}");
        let resp: serde_json::Value = self
            .http
            .get(&url)
            .send()
            .await?
            .json()
            .await
            .context("failed to parse order book")?;

        let mut total_depth = 0.0;
        if let Some(bids) = resp["bids"].as_array() {
            for bid in bids {
                let price = bid["price"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let size = bid["size"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                total_depth += price * size;
            }
        }
        Ok(total_depth)
    }

    pub async fn fetch_active_markets(&self) -> Result<Vec<GammaMarket>> {
        let mut all = Vec::new();
        let mut offset = 0;
        let page_size = 100;

        loop {
            let url = format!(
                "{GAMMA_API}/markets?closed=false&active=true&limit={page_size}&offset={offset}&order=volumeNum&ascending=false"
            );
            let resp = self.http.get(&url).send().await?;
            let text = resp.text().await?;
            let page: Vec<GammaMarket> = serde_json::from_str(&text).with_context(|| {
                format!(
                    "failed to parse gamma markets (first 200 chars): {}",
                    &text[..text.len().min(200)]
                )
            })?;

            let count = page.len();
            all.extend(page);

            if count < page_size || all.len() >= self.cfg.max_markets_fetch {
                break;
            }
            offset += count;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Ok(all)
    }

    pub async fn fetch_price_history(&self, token_id: &str) -> Result<Vec<PriceTick>> {
        let url = format!("{CLOB_API}/prices-history?market={token_id}&interval=max");
        #[derive(serde::Deserialize)]
        struct Resp {
            history: Vec<PriceTick>,
        }
        let resp: Resp = self.http.get(&url).send().await?.json().await?;
        Ok(resp.history)
    }

    fn expires_within_window(&self, end_date: Option<&str>) -> bool {
        let Some(date_str) = end_date else {
            return false;
        };
        let Ok(end) = date_str.parse::<DateTime<Utc>>() else {
            return false;
        };
        let now = Utc::now();
        let deadline = now + ChronoDuration::days(self.cfg.max_days_to_expiry);
        end > now && end <= deadline
    }

    fn get_yes_price(gm: &GammaMarket) -> Option<f64> {
        let s = gm.outcome_prices.as_ref()?;
        let p: Vec<String> = serde_json::from_str(s).ok()?;
        p.first().and_then(|s| s.parse::<f64>().ok())
    }

    /// Log an LLM estimate to the database for calibration tracking.
    #[allow(clippy::too_many_arguments)]
    async fn log_estimate(
        &self,
        market: &GammaMarket,
        role: &str,
        raw_prob: f64,
        raw_conf: f64,
        consensus_prob: Option<f64>,
        consensus_conf: Option<f64>,
        current_price: f64,
    ) {
        let result = sqlx::query(
            "INSERT INTO llm_estimates \
             (market_id, question, agent_role, raw_probability, raw_confidence, \
              consensus_probability, consensus_confidence, current_price) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(&market.market_id)
        .bind(&market.question)
        .bind(role)
        .bind(raw_prob)
        .bind(raw_conf)
        .bind(consensus_prob)
        .bind(consensus_conf)
        .bind(current_price)
        .execute(&self.pool)
        .await;

        if let Err(e) = result {
            tracing::warn!(err = %e, "Failed to log LLM estimate");
        }
    }

    /// Build the shared prompt for all agents (factual data only, no role framing).
    fn build_market_prompt(
        market: &GammaMarket,
        current_price: f64,
        news: &[NewsItem],
        history_summary: &str,
        past_bets: &str,
        calibration_summary: &str,
    ) -> String {
        let expiry = market.end_date.as_deref().unwrap_or("Unknown");
        let now = Utc::now();

        let mut news_block = String::new();
        for (i, item) in news.iter().enumerate() {
            let age = item
                .published
                .map(|p| {
                    let mins = (now - p).num_minutes();
                    if mins < 60 {
                        format!("{mins}min ago")
                    } else {
                        format!("{}h ago", mins / 60)
                    }
                })
                .unwrap_or_else(|| "recent".to_string());
            news_block.push_str(&format!(
                "{}. [{}] ({}) {}\n",
                i + 1,
                item.source,
                age,
                item.title
            ));
            if !item.summary.is_empty() {
                news_block.push_str(&format!("   {}\n", item.summary));
            }
        }

        let mut prompt = format!(
            "PREDICTION MARKET:\n\
             Question: \"{question}\"\n\
             Current YES price: {current_price:.4} ({pct:.1}%)\n\
             Expiry: {expiry}\n\
             Now: {now}\n\n\
             PRICE HISTORY:\n{history_summary}\n\n\
             RECENT NEWS:\n{news_block}\n\
             Analyze: Does any of this news change the probability of YES?\n\
             Is the current price already reflecting this news, or is there an edge?",
            question = market.question,
            pct = current_price * 100.0,
        );

        if !past_bets.is_empty() {
            prompt.push_str(&format!(
                "\n\nPAST BET ANALYSIS (learn from our mistakes):\n{past_bets}"
            ));
        }

        if !calibration_summary.is_empty() {
            prompt.push_str(&format!("\n\n{calibration_summary}"));
        }

        prompt
    }

    /// Multi-agent news impact assessment with consensus aggregation.
    async fn assess_news_impact(
        &self,
        market: &GammaMarket,
        current_price: f64,
        news: &[NewsItem],
        history_summary: &str,
        past_bets: &str,
    ) -> Result<(f64, f64, String)> {
        let cal_summary = self.calibration.read().await.summary();
        let prompt = Self::build_market_prompt(
            market,
            current_price,
            news,
            history_summary,
            past_bets,
            &cal_summary,
        );

        let roles = AgentRole::roles_for(self.cfg.consensus_agents);
        let mut assessments: Vec<(AgentRole, f64, f64, String)> = Vec::new();

        for (i, &role) in roles.iter().enumerate() {
            if i > 0 {
                // 3 RPM = 1 call per 20s, add 1s safety margin
                tracing::info!("Waiting 21s for rate limit...");
                tokio::time::sleep(Duration::from_secs(21)).await;
            }

            let agent = self.build_agent(role);
            let result = agent
                .chat(prompt.as_str(), vec![])
                .await
                .map_err(|e| anyhow::anyhow!("LLM call ({}) failed: {e}", role.label()))
                .and_then(|resp| parse_llm_response(&resp));

            match result {
                Ok((prob, conf, reasoning)) => {
                    tracing::info!(
                        market = %market.question,
                        agent = role.label(),
                        prob = format_args!("{:.1}%", prob * 100.0),
                        conf = format_args!("{:.0}%", conf * 100.0),
                        "{reasoning}",
                    );

                    self.log_estimate(market, role.label(), prob, conf, None, None, current_price)
                        .await;

                    assessments.push((role, prob, conf, reasoning));
                }
                Err(e) => {
                    tracing::warn!(
                        market = %market.question,
                        agent = role.label(),
                        err = %e,
                        "Agent assessment failed, continuing with remaining agents"
                    );
                }
            }
        }

        if assessments.is_empty() {
            anyhow::bail!("All agents failed for market: {}", market.question);
        }

        // Aggregate consensus
        let (consensus_prob, consensus_conf, consensus_reasoning) =
            aggregate_assessments(&assessments, self.cfg.consensus_max_spread);

        // Apply calibration correction
        let calibrated_prob = self.calibration.read().await.correct(consensus_prob);

        if (calibrated_prob - consensus_prob).abs() > 0.02 {
            tracing::info!(
                raw = format_args!("{:.1}%", consensus_prob * 100.0),
                calibrated = format_args!("{:.1}%", calibrated_prob * 100.0),
                "Calibration correction applied"
            );
        }

        // Log consensus estimate
        self.log_estimate(
            market,
            "consensus",
            consensus_prob,
            consensus_conf,
            Some(calibrated_prob),
            Some(consensus_conf),
            current_price,
        )
        .await;

        tracing::info!(
            market = %market.question,
            prob = format_args!("{:.1}%", calibrated_prob * 100.0),
            conf = format_args!("{:.0}%", consensus_conf * 100.0),
            "Consensus assessment"
        );

        Ok((calibrated_prob, consensus_conf, consensus_reasoning))
    }

    /// Main scan: fetch news → match to markets → LLM assesses impact → signals.
    /// `seen_headlines` tracks already-processed headlines across cycles to avoid
    /// wasting LLM calls on stale news.
    pub async fn scan(
        &self,
        skip_market_ids: &[String],
        past_bets_summary: &str,
        seen_headlines: &mut std::collections::HashSet<String>,
    ) -> Result<ScanResult> {
        // Step 1: Fetch all active markets (ALL categories, not just crypto)
        let markets = self.fetch_active_markets().await?;
        tracing::info!(total = markets.len(), "Fetched active markets");

        let eligible: Vec<GammaMarket> = markets
            .into_iter()
            .filter(|m| m.volume_num >= self.cfg.min_volume)
            .filter(|m| m.yes_token_id().is_some())
            .filter(|m| self.expires_within_window(m.end_date.as_deref()))
            .filter(|m| !skip_market_ids.contains(&m.market_id))
            .filter(|m| {
                let price = Self::get_yes_price(m).unwrap_or(0.0);
                price > self.cfg.min_price && price < self.cfg.max_price
            })
            .collect();

        tracing::info!(
            eligible = eligible.len(),
            "Eligible markets (vol>${}, ≤{}d expiry)",
            self.cfg.min_volume,
            self.cfg.max_days_to_expiry,
        );

        let markets_scanned = eligible.len();

        // Step 2: Fetch breaking news from all sources
        let mut news = self.news.fetch_all().await;
        let news_total = news.len();

        // Filter out headlines we've already processed in previous cycles
        let before = news.len();
        news.retain(|item| {
            let key = item
                .title
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .take(60)
                .collect::<String>();
            !seen_headlines.contains(&key)
        });
        if before != news.len() {
            tracing::info!(
                total = before,
                new = news.len(),
                skipped = before - news.len(),
                "Filtered already-seen headlines"
            );
        }

        // Remember these headlines for next cycle
        for item in &news {
            let key = item
                .title
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .take(60)
                .collect::<String>();
            seen_headlines.insert(key);
        }

        // Prune seen set if it grows too large (keep last ~2000)
        if seen_headlines.len() > 3000 {
            let drain_count = seen_headlines.len() - 2000;
            let keys: Vec<String> = seen_headlines.iter().take(drain_count).cloned().collect();
            for k in keys {
                seen_headlines.remove(&k);
            }
        }

        let news_new = news.len();

        if news.is_empty() {
            tracing::warn!("No news fetched — check network/API access");
            return Ok(ScanResult {
                signals: Vec::new(),
                rejections: Vec::new(),
                markets_scanned,
                news_total,
                news_new: 0,
            });
        }

        // Step 3: Match news to markets by keyword relevance
        let matches = NewsAggregator::match_to_markets(&news, &eligible);

        tracing::info!(
            matched = matches.len(),
            "Markets matched with relevant news"
        );

        if matches.is_empty() {
            return Ok(ScanResult {
                signals: Vec::new(),
                rejections: Vec::new(),
                markets_scanned,
                news_total,
                news_new,
            });
        }

        // Step 4: For top matches, check book depth and get price history
        let mut candidates: Vec<(NewsMatch, f64, String, f64)> = Vec::new();

        for nm in matches.iter().take(self.cfg.max_llm_candidates * 2) {
            let token_id = nm.market.yes_token_id().unwrap();
            let current_price = Self::get_yes_price(&nm.market).unwrap_or(0.0);

            // Check liquidity
            tokio::time::sleep(Duration::from_millis(100)).await;
            let book_depth = self.fetch_book_depth(&token_id).await.unwrap_or(0.0);

            if book_depth < self.cfg.min_book_depth {
                tracing::debug!(
                    market = %nm.market.question,
                    depth = format_args!("${book_depth:.0}"),
                    "Skipping — thin book"
                );
                continue;
            }

            // Get price history for context
            tokio::time::sleep(Duration::from_millis(100)).await;
            let history = self
                .fetch_price_history(&token_id)
                .await
                .unwrap_or_default();
            let history_summary = summarize_history(&history, current_price);

            candidates.push((nm.clone(), current_price, history_summary, book_depth));

            if candidates.len() >= self.cfg.max_llm_candidates {
                break;
            }
        }

        if candidates.is_empty() {
            tracing::info!("No news-matched markets passed liquidity filter");
            return Ok(ScanResult {
                signals: Vec::new(),
                rejections: Vec::new(),
                markets_scanned,
                news_total,
                news_new,
            });
        }

        tracing::info!(
            count = candidates.len(),
            "Assessing news impact with LLM..."
        );

        // Step 5: LLM assesses news impact on each candidate
        let mut signals = Vec::new();
        let mut rejections = Vec::new();

        for (i, (nm, current_price, history_summary, book_depth)) in candidates.iter().enumerate() {
            // Rate limit between candidates: just need 21s since last LLM call.
            // The previous candidate's last agent call already waited, so only
            // the gap between that call finishing and the next one starting matters.
            if i > 0 {
                tracing::info!("Waiting 21s for rate limit between candidates...");
                tokio::time::sleep(Duration::from_secs(21)).await;
            }

            let (prob, confidence, reasoning) = match self
                .assess_news_impact(
                    &nm.market,
                    *current_price,
                    &nm.news,
                    history_summary,
                    past_bets_summary,
                )
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(market = %nm.market.market_id, err = %e, "LLM assessment failed");
                    continue;
                }
            };

            // Determine best side
            let yes_edge = prob - current_price;
            let no_price = 1.0 - current_price;
            let no_prob = 1.0 - prob;
            let no_edge = no_prob - no_price;

            let (side, edge, bet_price, bet_prob) = if yes_edge >= no_edge && yes_edge > 0.0 {
                (BetSide::Yes, yes_edge, *current_price, prob)
            } else if no_edge > 0.0 {
                (BetSide::No, no_edge, no_price, no_prob)
            } else {
                let reason = format!("no edge (prob {:.0}% ~ price {:.0}%)", prob * 100.0, current_price * 100.0);
                tracing::info!(
                    market = %nm.market.question,
                    prob = format_args!("{:.1}%", prob * 100.0),
                    price = format_args!("{:.1}%", current_price * 100.0),
                    "No edge on either side"
                );
                rejections.push(RejectedSignal { question: nm.market.question.clone(), reason });
                continue;
            };

            // Use full Kelly here; each strategy will scale by its own fraction
            let kelly_size = fractional_kelly(bet_prob, bet_price, 1.0);
            let effective_edge = edge * confidence;

            let news_titles: Vec<&str> = nm.news.iter().map(|n| n.title.as_str()).collect();
            tracing::info!(
                market = %nm.market.question,
                side = %side,
                price = format_args!("{:.1}%", bet_price * 100.0),
                prob = format_args!("{:.1}%", prob * 100.0),
                edge = format_args!("+{:.1}%", edge * 100.0),
                eff_edge = format_args!("+{:.1}%", effective_edge * 100.0),
                conf = format_args!("{:.0}%", confidence * 100.0),
                news = ?news_titles,
                "Signal candidate"
            );

            // Use permissive gate; individual strategies apply their own thresholds
            if effective_edge < 0.03 {
                let reason = format!("edge {:.1}% < 3%", effective_edge * 100.0);
                tracing::info!(
                    market = %nm.market.question,
                    eff_edge = format_args!("+{:.1}%", effective_edge * 100.0),
                    "REJECTED: effective edge below 3% gate"
                );
                rejections.push(RejectedSignal { question: nm.market.question.clone(), reason });
                continue;
            }
            if kelly_size <= 0.005 {
                let reason = format!("kelly {:.3} < 0.005", kelly_size);
                tracing::info!(
                    market = %nm.market.question,
                    kelly = format_args!("{:.3}", kelly_size),
                    "REJECTED: full Kelly too small"
                );
                rejections.push(RejectedSignal { question: nm.market.question.clone(), reason });
                continue;
            }
            if confidence < 0.30 {
                let reason = format!("conf {:.0}% < 30%", confidence * 100.0);
                tracing::info!(
                    market = %nm.market.question,
                    conf = format_args!("{:.0}%", confidence * 100.0),
                    "REJECTED: confidence below 30% gate"
                );
                rejections.push(RejectedSignal { question: nm.market.question.clone(), reason });
                continue;
            }

            tracing::info!(
                market = %nm.market.question,
                side = %side,
                eff_edge = format_args!("+{:.1}%", effective_edge * 100.0),
                kelly = format_args!("{:.1}%", kelly_size * 100.0),
                conf = format_args!("{:.0}%", confidence * 100.0),
                "ACCEPTED as signal — passing to strategies"
            );

            let news_headlines: Vec<String> = nm.news.iter().map(|n| n.title.clone()).collect();
            signals.push(Signal {
                market_id: nm.market.market_id.clone(),
                question: nm.market.question.clone(),
                side,
                current_price: bet_price,
                estimated_prob: bet_prob,
                confidence,
                edge,
                kelly_size,
                reasoning,
                end_date: nm.market.end_date.clone(),
                volume: nm.market.volume_num,
                context: BetContext {
                    btc_price: 0.0,
                    eth_price: 0.0,
                    sol_price: 0.0,
                    btc_24h_change: 0.0,
                    btc_funding_rate: 0.0,
                    btc_open_interest: 0.0,
                    fear_greed: String::new(),
                    book_depth: *book_depth,
                    news_headlines,
                },
            });
        }

        signals.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap());

        tracing::info!(signals = signals.len(), rejections = rejections.len(), "Final news-driven signals");
        Ok(ScanResult {
            signals,
            rejections,
            markets_scanned,
            news_total,
            news_new,
        })
    }
}

/// Aggregate multi-agent assessments into a consensus probability, confidence, and reasoning.
fn aggregate_assessments(
    assessments: &[(AgentRole, f64, f64, String)],
    max_spread: f64,
) -> (f64, f64, String) {
    if assessments.len() == 1 {
        let (role, prob, conf, reasoning) = &assessments[0];
        return (*prob, *conf, format!("[{}] {}", role.label(), reasoning));
    }

    // Confidence-weighted average probability
    let total_weight: f64 = assessments.iter().map(|(_, _, c, _)| c).sum();
    let weighted_prob = if total_weight > 0.0 {
        assessments.iter().map(|(_, p, c, _)| p * c).sum::<f64>() / total_weight
    } else {
        assessments.iter().map(|(_, p, _, _)| p).sum::<f64>() / assessments.len() as f64
    };

    // Measure disagreement: standard deviation of probabilities
    let mean_prob =
        assessments.iter().map(|(_, p, _, _)| p).sum::<f64>() / assessments.len() as f64;
    let variance = assessments
        .iter()
        .map(|(_, p, _, _)| (p - mean_prob).powi(2))
        .sum::<f64>()
        / assessments.len() as f64;
    let std_dev = variance.sqrt();

    // Agreement factor: 1.0 when agents agree, drops toward 0 as they diverge
    let agreement = if max_spread > 0.0 {
        (1.0 - std_dev / max_spread).max(0.0)
    } else if std_dev < 0.001 {
        1.0
    } else {
        0.0
    };

    // Final confidence: minimum confidence * agreement factor
    let min_conf = assessments
        .iter()
        .map(|(_, _, c, _)| *c)
        .fold(f64::MAX, f64::min);
    let consensus_conf = (min_conf * agreement).clamp(0.0, 1.0);

    // Build combined reasoning
    let mut reasoning_parts: Vec<String> = assessments
        .iter()
        .map(|(role, prob, conf, reasoning)| {
            format!(
                "[{} {:.0}%@{:.0}%] {}",
                role.label(),
                prob * 100.0,
                conf * 100.0,
                reasoning
            )
        })
        .collect();
    reasoning_parts.push(format!(
        "Consensus: {:.1}% (spread={:.1}%, agreement={:.0}%)",
        weighted_prob * 100.0,
        std_dev * 100.0,
        agreement * 100.0,
    ));
    let consensus_reasoning = reasoning_parts.join(" | ");

    (weighted_prob, consensus_conf, consensus_reasoning)
}

fn summarize_history(history: &[PriceTick], current_price: f64) -> String {
    if history.len() < 3 {
        return format!(
            "Current price: {:.1}%, minimal history",
            current_price * 100.0
        );
    }

    let prices: Vec<f64> = history.iter().map(|t| t.p).collect();
    let recent: Vec<String> = prices
        .iter()
        .rev()
        .take(10)
        .rev()
        .map(|p| format!("{:.2}", p))
        .collect();
    let min = prices.iter().cloned().fold(f64::MAX, f64::min);
    let max = prices.iter().cloned().fold(f64::MIN, f64::max);
    let avg = prices.iter().sum::<f64>() / prices.len() as f64;

    // Recent movement (last 5 ticks vs 5 before that)
    let n = prices.len();
    let recent_avg = if n >= 5 {
        prices[n - 5..].iter().sum::<f64>() / 5.0
    } else {
        avg
    };
    let older_avg = if n >= 10 {
        prices[n - 10..n - 5].iter().sum::<f64>() / 5.0
    } else {
        avg
    };
    let momentum = recent_avg - older_avg;

    format!(
        "Recent prices: [{}]\nRange: {min:.2}-{max:.2}, avg={avg:.2}\nMomentum: {:+.3} (recent vs prior)\nTicks: {n}",
        recent.join(", "),
        momentum,
    )
}

fn parse_llm_response(response: &str) -> Result<(f64, f64, String)> {
    let json_str = if let Some(start) = response.find('{') {
        if let Some(end) = response[start..].find('}') {
            &response[start..=start + end]
        } else {
            response
        }
    } else {
        response
    };

    #[derive(serde::Deserialize)]
    struct Est {
        probability: f64,
        #[serde(default = "default_confidence")]
        confidence: f64,
        #[serde(default)]
        reasoning: String,
    }
    fn default_confidence() -> f64 {
        0.5
    }

    let est: Est = serde_json::from_str(json_str).context("failed to parse LLM response")?;
    Ok((
        est.probability.clamp(0.01, 0.99),
        est.confidence.clamp(0.0, 1.0),
        est.reasoning,
    ))
}

/// Strip characters that break Telegram MarkdownV1 inside italic `_..._` blocks.
fn sanitize_markdown(s: &str) -> String {
    s.replace(['_', '*'], " ")
        .replace('`', "'")
        .replace('[', "(")
        .replace(']', ")")
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AgentRole ---

    #[test]
    fn test_roles_for_clamped() {
        assert_eq!(AgentRole::roles_for(0).len(), 1);
        assert_eq!(AgentRole::roles_for(1).len(), 1);
        assert_eq!(AgentRole::roles_for(2).len(), 2);
        assert_eq!(AgentRole::roles_for(3).len(), 3);
        assert_eq!(AgentRole::roles_for(99).len(), 3);
    }

    #[test]
    fn test_roles_for_order() {
        let roles = AgentRole::roles_for(3);
        assert_eq!(roles[0].label(), "skeptic");
        assert_eq!(roles[1].label(), "catalyst");
        assert_eq!(roles[2].label(), "base_rate");
    }

    // --- aggregate_assessments ---

    fn make_assessment(role: AgentRole, prob: f64, conf: f64) -> (AgentRole, f64, f64, String) {
        (role, prob, conf, format!("test {}", role.label()))
    }

    #[test]
    fn test_single_agent_passthrough() {
        let a = vec![make_assessment(AgentRole::Skeptic, 0.7, 0.8)];
        let (prob, conf, reasoning) = aggregate_assessments(&a, 0.15);
        assert!((prob - 0.7).abs() < 1e-9);
        assert!((conf - 0.8).abs() < 1e-9);
        assert!(reasoning.contains("skeptic"));
    }

    #[test]
    fn test_two_agents_agree() {
        let a = vec![
            make_assessment(AgentRole::Skeptic, 0.70, 0.8),
            make_assessment(AgentRole::Catalyst, 0.72, 0.9),
        ];
        let (prob, conf, reasoning) = aggregate_assessments(&a, 0.15);
        // Probability should be confidence-weighted average
        let expected_prob = (0.70 * 0.8 + 0.72 * 0.9) / (0.8 + 0.9);
        assert!((prob - expected_prob).abs() < 1e-9);
        // Spread is tiny (0.01), agreement should be high
        assert!(conf > 0.7, "conf={conf} should be > 0.7 when agents agree");
        assert!(reasoning.contains("Consensus"));
    }

    #[test]
    fn test_two_agents_disagree_kills_confidence() {
        let a = vec![
            make_assessment(AgentRole::Skeptic, 0.30, 0.8),
            make_assessment(AgentRole::Catalyst, 0.80, 0.9),
        ];
        let (_prob, conf, _reasoning) = aggregate_assessments(&a, 0.15);
        // Spread is 0.25 which exceeds max_spread of 0.15 → agreement=0 → conf=0
        assert!(
            conf < 0.01,
            "conf={conf} should be ~0 when agents wildly disagree"
        );
    }

    #[test]
    fn test_zero_confidence_agents() {
        let a = vec![
            make_assessment(AgentRole::Skeptic, 0.50, 0.0),
            make_assessment(AgentRole::Catalyst, 0.60, 0.0),
        ];
        let (prob, conf, _) = aggregate_assessments(&a, 0.15);
        // With 0 total weight, falls back to simple average
        assert!((prob - 0.55).abs() < 1e-9);
        // min_conf is 0 → consensus_conf is 0
        assert!(conf < 1e-9);
    }

    #[test]
    fn test_max_spread_zero() {
        let a = vec![
            make_assessment(AgentRole::Skeptic, 0.60, 0.8),
            make_assessment(AgentRole::Catalyst, 0.60, 0.8),
        ];
        let (_, conf, _) = aggregate_assessments(&a, 0.0);
        // Identical probabilities → std_dev ≈ 0 → agreement = 1.0
        assert!(
            conf > 0.7,
            "conf={conf} should be high when agents agree exactly"
        );

        // Now with disagreement
        let a2 = vec![
            make_assessment(AgentRole::Skeptic, 0.40, 0.8),
            make_assessment(AgentRole::Catalyst, 0.60, 0.8),
        ];
        let (_, conf2, _) = aggregate_assessments(&a2, 0.0);
        assert!(
            conf2 < 0.01,
            "conf={conf2} should be 0 when spread>0 and max_spread=0"
        );
    }

    #[test]
    fn test_three_agents_weighted_avg() {
        let a = vec![
            make_assessment(AgentRole::Skeptic, 0.50, 0.5),
            make_assessment(AgentRole::Catalyst, 0.70, 1.0),
            make_assessment(AgentRole::BaseRate, 0.60, 0.5),
        ];
        let (prob, _, _) = aggregate_assessments(&a, 0.15);
        let expected = (0.50 * 0.5 + 0.70 * 1.0 + 0.60 * 0.5) / (0.5 + 1.0 + 0.5);
        assert!(
            (prob - expected).abs() < 1e-9,
            "prob={prob} expected={expected}"
        );
    }

    // --- parse_llm_response ---

    #[test]
    fn test_parse_clean_json() {
        let input = r#"{"probability": 0.75, "confidence": 0.8, "reasoning": "strong signal"}"#;
        let (prob, conf, reason) = parse_llm_response(input).unwrap();
        assert!((prob - 0.75).abs() < 1e-9);
        assert!((conf - 0.8).abs() < 1e-9);
        assert_eq!(reason, "strong signal");
    }

    #[test]
    fn test_parse_json_with_surrounding_text() {
        let input = "Here is my analysis:\n{\"probability\": 0.65, \"confidence\": 0.4, \"reasoning\": \"weak\"}\nThat's all.";
        let (prob, conf, _) = parse_llm_response(input).unwrap();
        assert!((prob - 0.65).abs() < 1e-9);
        assert!((conf - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_parse_clamps_extreme_values() {
        let input = r#"{"probability": 1.5, "confidence": -0.5, "reasoning": "extreme"}"#;
        let (prob, conf, _) = parse_llm_response(input).unwrap();
        assert!((prob - 0.99).abs() < 1e-9);
        assert!(conf < 1e-9);
    }

    #[test]
    fn test_parse_missing_confidence_defaults() {
        let input = r#"{"probability": 0.6}"#;
        let (prob, conf, _) = parse_llm_response(input).unwrap();
        assert!((prob - 0.6).abs() < 1e-9);
        assert!((conf - 0.5).abs() < 1e-9); // default_confidence
    }

    #[test]
    fn test_parse_invalid_json_fails() {
        assert!(parse_llm_response("not json at all").is_err());
    }

    // --- sanitize_markdown ---

    #[test]
    fn test_sanitize_strips_markdown_chars() {
        let input = "BTC_price *surged* to `$100k` [link]";
        let result = sanitize_markdown(input);
        assert!(!result.contains('_'));
        assert!(!result.contains('*'));
        assert!(!result.contains('`'));
        assert!(!result.contains('['));
        assert!(!result.contains(']'));
    }

    #[test]
    fn test_sanitize_preserves_normal_text() {
        let input = "Bitcoin rose 5% today";
        assert_eq!(sanitize_markdown(input), input);
    }

    // --- summarize_history ---

    #[test]
    fn test_summarize_history_minimal() {
        let result = summarize_history(&[], 0.5);
        assert!(result.contains("minimal history"));
    }

    #[test]
    fn test_summarize_history_with_data() {
        let ticks: Vec<crate::data::models::PriceTick> = (0..20)
            .map(|i| crate::data::models::PriceTick {
                t: 1000 + i,
                p: 0.4 + (i as f64) * 0.01,
            })
            .collect();
        let result = summarize_history(&ticks, 0.59);
        assert!(result.contains("Recent prices:"));
        assert!(result.contains("Momentum:"));
        assert!(result.contains("Ticks: 20"));
    }

    // --- Signal::score ---

    #[test]
    fn test_signal_score() {
        let signal = Signal {
            market_id: "test".into(),
            question: "Test?".into(),
            side: BetSide::Yes,
            current_price: 0.50,
            estimated_prob: 0.70,
            confidence: 0.80,
            edge: 0.20,
            kelly_size: 0.10,
            reasoning: "test".into(),
            end_date: None,
            volume: 1000.0,
            context: BetContext::default(),
        };
        let expected = 0.20 * 0.80 * 0.10;
        assert!((signal.score() - expected).abs() < 1e-9);
    }

    // --- parse edge cases ---

    #[test]
    fn test_parse_nested_json() {
        // LLM sometimes wraps in markdown code blocks
        let input =
            "```json\n{\"probability\": 0.55, \"confidence\": 0.6, \"reasoning\": \"test\"}\n```";
        let (prob, _, _) = parse_llm_response(input).unwrap();
        assert!((prob - 0.55).abs() < 1e-9);
    }

    #[test]
    fn test_parse_probability_boundaries() {
        // probability of exactly 0.0 should clamp to 0.01
        let input = r#"{"probability": 0.0, "confidence": 0.5, "reasoning": "no chance"}"#;
        let (prob, _, _) = parse_llm_response(input).unwrap();
        assert!((prob - 0.01).abs() < 1e-9);
    }
}
