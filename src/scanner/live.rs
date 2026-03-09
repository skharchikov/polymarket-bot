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

use crate::bayesian::{self, AgentAssessment, BayesianEstimate};
use crate::calibration::CalibrationCurve;
use crate::config::AppConfig;
use crate::data::models::{GammaMarket, PriceTick};
use crate::model::features::MarketFeatures;
use crate::model::xgb::XgbModel;
use crate::pricing::kelly::fractional_kelly;
use crate::storage::portfolio::{BetContext, BetSide};

use super::news::{NewsAggregator, NewsItem, NewsMatch};

const GAMMA_API: &str = "https://gamma-api.polymarket.com";
const CLOB_API: &str = "https://clob.polymarket.com";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalSource {
    XgBoost,
    LlmConsensus,
}

impl SignalSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::XgBoost => "XGBoost Model",
            Self::LlmConsensus => "LLM Consensus",
        }
    }
}

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
    pub polymarket_url: String,
    /// Bayesian prior (market price) before update
    pub prior: f64,
    /// Combined likelihood ratio from all agents
    pub combined_lr: f64,
    /// Number of news items matched to this market
    pub news_matched_count: usize,
    pub context: BetContext,
    /// Whether this signal came from XGBoost or LLM
    pub source: SignalSource,
}

impl Signal {
    pub fn to_telegram_message(&self) -> String {
        let (emoji, side_label) = match self.side {
            BetSide::Yes => ("\u{1f7e2}", "YES"),
            BetSide::No => ("\u{1f534}", "NO"),
        };

        let source_tag = match self.source {
            SignalSource::XgBoost => "\u{1f916} XGBoost",
            SignalSource::LlmConsensus => "\u{1f9e0} LLM",
        };

        // Format reasoning block based on source
        let reasoning_block = if self.source == SignalSource::XgBoost {
            format!(
                "\n\u{1f916} _Model: {:.1}% vs market {:.1}%_",
                self.estimated_prob * 100.0,
                self.prior * 100.0
            )
        } else {
            let reasoning_lines: Vec<&str> = self.reasoning.split(" | ").collect();
            if reasoning_lines.len() > 1 {
                let mut block = String::from("\n\u{1f9e0} *Bayesian Agents:*\n");
                for line in &reasoning_lines {
                    let safe = sanitize_markdown(line);
                    block.push_str(&format!("  \u{2022} _{safe}_\n"));
                }
                block
            } else {
                format!("\n\u{1f4a1} _{}_", sanitize_markdown(&self.reasoning))
            }
        };

        format!(
            "{emoji} *{side_label} Signal* ({source})\n\n\
             \u{1f4cb} [{question}]({url})\n\n\
             \u{1f4b0} Market price: `{price:.1}\u{00a2}`\n\
             \u{1f9ea} Estimate: `{prior:.1}%` \u{2192} `{posterior:.1}%` (LR=`{lr:.2}`)\n\
             \u{1f4ca} Edge: `+{edge:.1}%`\n\
             \u{1f512} Confidence: `{conf:.0}%`\n\
             \u{1f4d0} Kelly size: `{kelly:.1}%` of bankroll\n\
             \u{1f4a7} Volume: `${vol:.0}`\n\
             \u{1f4f0} News matched: {news_count} articles\n\
             \u{23f0} Expires: {end}\
             {reasoning}",
            source = source_tag,
            question = self.question,
            url = self.polymarket_url,
            price = self.current_price * 100.0,
            prior = self.prior * 100.0,
            posterior = self.estimated_prob * 100.0,
            lr = self.combined_lr,
            edge = self.edge * 100.0,
            conf = self.confidence * 100.0,
            kelly = self.kelly_size * 100.0,
            vol = self.volume,
            news_count = self.news_matched_count,
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
    pub news_matched: usize,
    pub llm_assessed: usize,
    /// Per-source news counts: (source_name, count)
    pub source_counts: Vec<(String, usize)>,
}

/// A signal that was evaluated by the LLM but didn't pass the scanner gate.
#[derive(Debug, Clone)]
pub struct RejectedSignal {
    pub market_id: String,
    pub question: String,
    pub reason: String,
    pub current_price: Option<f64>,
    pub estimated_prob: Option<f64>,
    pub edge: Option<f64>,
    pub confidence: Option<f64>,
    pub combined_lr: Option<f64>,
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
                "You are a SKEPTICAL prediction market analyst using Bayesian reasoning.\n\n\
                 Your default assumption: the market price is already correct. News is usually \
                 already priced in. Your job is to find reasons why the current price is RIGHT \
                 and push back against overreaction to headlines.\n\n\
                 Only give high confidence (>0.7) when you find genuinely new, market-moving \
                 information that clearly hasn't been absorbed.\n\n\
                 Respond with ONLY a JSON object:\n\
                 {\"likelihood_ratio\": X.XX, \"confidence\": 0.XX, \"reasoning\": \"one sentence\"}\n\n\
                 - likelihood_ratio: how much more likely is this news in a world where YES happens \
                   vs a world where NO happens. Examples:\n\
                   * 1.0 = news is equally likely either way (uninformative)\n\
                   * 2.0 = news is 2x more likely if YES happens (moderate YES evidence)\n\
                   * 0.5 = news is 2x more likely if NO happens (moderate NO evidence)\n\
                   * Range: 0.1 to 10.0\n\
                 - confidence: 0.0 (no useful info) to 1.0 (highly certain about your LR estimate)\n\
                 - reasoning: explain why the market IS or ISN'T already priced correctly"
            }
            Self::Catalyst => {
                "You are an AGGRESSIVE prediction market analyst using Bayesian reasoning.\n\n\
                 Your job: find fresh information that the market hasn't absorbed yet. \
                 Look for breaking news, policy changes, data releases, and events that \
                 clearly shift probabilities.\n\n\
                 Think in terms of evidence strength:\n\
                 - Would you expect to see this news MORE in worlds where YES happens, or NO?\n\
                 - How much more? That ratio is your likelihood ratio.\n\n\
                 Be calibrated, but don't be afraid to express conviction when evidence is strong.\n\n\
                 Respond with ONLY a JSON object:\n\
                 {\"likelihood_ratio\": X.XX, \"confidence\": 0.XX, \"reasoning\": \"one sentence\"}\n\n\
                 - likelihood_ratio: P(news | YES) / P(news | NO). Examples:\n\
                   * 1.0 = uninformative\n\
                   * 3.0 = strong YES evidence (news 3x more likely in YES-worlds)\n\
                   * 0.3 = strong NO evidence (news 3x more likely in NO-worlds)\n\
                   * Range: 0.1 to 10.0\n\
                 - confidence: 0.0 (no useful info) to 1.0 (highly certain about your LR estimate)\n\
                 - reasoning: explain what catalyst matters and why"
            }
            Self::BaseRate => {
                "You are a STATISTICAL prediction market analyst using Bayesian reasoning.\n\n\
                 Ignore narratives and hype. Focus on:\n\
                 1. Historical base rates: how often do events like this happen?\n\
                 2. Price history momentum: is the trend meaningful or noise?\n\
                 3. Time to expiry: how much can the probability realistically move?\n\n\
                 Estimate the likelihood ratio from a purely statistical perspective.\n\n\
                 Respond with ONLY a JSON object:\n\
                 {\"likelihood_ratio\": X.XX, \"confidence\": 0.XX, \"reasoning\": \"one sentence\"}\n\n\
                 - likelihood_ratio: P(news | YES) / P(news | NO) based on base rates. Examples:\n\
                   * 1.0 = news doesn't statistically change anything\n\
                   * 1.5 = mild statistical evidence for YES\n\
                   * 0.7 = mild statistical evidence for NO\n\
                   * Range: 0.1 to 10.0\n\
                 - confidence: 0.0 (no useful info) to 1.0 (highly certain about your LR estimate)\n\
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

const MODEL_PATH: &str = "model/xgb_model.json";

pub struct LiveScanner {
    http: Client,
    openai_client: openai::Client,
    news: NewsAggregator,
    pool: PgPool,
    calibration: RwLock<CalibrationCurve>,
    cfg: Arc<AppConfig>,
    xgb_model: Option<XgbModel>,
}

impl LiveScanner {
    pub async fn new(cfg: &Arc<AppConfig>, pool: PgPool) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client");
        let calibration = CalibrationCurve::load(&pool, cfg.calibration_min_samples).await?;

        // Load XGBoost model if available (trained by scripts/train_model.py)
        let model_path = std::path::Path::new(MODEL_PATH);
        let xgb_model = match XgbModel::load(model_path) {
            Ok(m) => {
                tracing::info!(trees = m.n_trees(), "XGBoost model loaded");
                Some(m)
            }
            Err(e) => {
                tracing::info!(err = %e, "No XGBoost model — using LLM only");
                None
            }
        };

        Ok(Self {
            news: NewsAggregator::new(http.clone()),
            http,
            openai_client: openai::Client::from_env(),
            pool,
            calibration: RwLock::new(calibration),
            cfg: Arc::clone(cfg),
            xgb_model,
        })
    }

    /// Reload calibration curve from DB (call periodically).
    pub async fn reload_calibration(&self) -> Result<()> {
        let curve = CalibrationCurve::load(&self.pool, self.cfg.calibration_min_samples).await?;
        *self.calibration.write().await = curve;
        Ok(())
    }

    /// Hot-reload XGBoost model from disk (call after retraining).
    #[allow(dead_code)]
    pub fn reload_model(&mut self) {
        let model_path = std::path::Path::new(MODEL_PATH);
        match XgbModel::load(model_path) {
            Ok(m) => {
                tracing::info!(trees = m.n_trees(), "XGBoost model reloaded");
                self.xgb_model = Some(m);
            }
            Err(e) => {
                tracing::debug!(err = %e, "Model reload skipped");
            }
        }
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

    /// Fetch current YES price for a market by ID.
    pub async fn fetch_current_price(&self, market_id: &str) -> Result<Option<f64>> {
        let url = format!("{GAMMA_API}/markets/{market_id}");
        let resp = self.http.get(&url).send().await?;
        let text = resp.text().await?;
        let market: GammaMarket = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse market {market_id}"))?;
        Ok(Self::get_yes_price(&market))
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
             Analyze: Does any of this news change the likelihood of YES?\n\
             Think about it as: in worlds where this event resolves YES, how likely \
             would you be to see this news? Compare to worlds where it resolves NO.\n\
             The current price reflects the market consensus — is there evidence to update it?",
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

    /// Multi-agent Bayesian news impact assessment.
    ///
    /// Each agent estimates a likelihood ratio (how much more likely is this news
    /// in YES-worlds vs NO-worlds). We start from the market price as prior and
    /// apply Bayes' rule sequentially.
    async fn assess_news_impact(
        &self,
        market: &GammaMarket,
        current_price: f64,
        news: &[NewsItem],
        history_summary: &str,
        past_bets: &str,
    ) -> Result<(f64, f64, f64, String)> {
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
        let mut assessments: Vec<AgentAssessment> = Vec::new();

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
                Ok(estimate) => {
                    let (lr, conf, reasoning) = match estimate {
                        LlmEstimate::LikelihoodRatio {
                            lr,
                            confidence,
                            reasoning,
                        } => (lr, confidence, reasoning),
                        LlmEstimate::RawProbability {
                            probability,
                            confidence,
                            reasoning,
                        } => {
                            // Fallback: convert raw probability to LR relative to market price
                            let lr = bayesian::prob_to_odds(probability)
                                / bayesian::prob_to_odds(current_price);
                            tracing::debug!(
                                agent = role.label(),
                                "Legacy probability response, converted to LR={lr:.2}"
                            );
                            (lr.clamp(0.1, 10.0), confidence, reasoning)
                        }
                    };

                    tracing::info!(
                        market = %market.question,
                        agent = role.label(),
                        lr = format_args!("{lr:.2}"),
                        conf = format_args!("{:.0}%", conf * 100.0),
                        "{reasoning}",
                    );

                    self.log_estimate(market, role.label(), lr, conf, None, None, current_price)
                        .await;

                    assessments.push(AgentAssessment {
                        role: role.label().to_string(),
                        likelihood_ratio: lr,
                        confidence: conf,
                        reasoning,
                    });
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

        // Bayesian update: market price as prior, agents provide evidence
        let estimate = bayesian::bayesian_update(current_price, &assessments);

        // Apply calibration correction to the posterior
        let calibrated_prob = self.calibration.read().await.correct(estimate.posterior);

        if (calibrated_prob - estimate.posterior).abs() > 0.02 {
            tracing::info!(
                raw = format_args!("{:.1}%", estimate.posterior * 100.0),
                calibrated = format_args!("{:.1}%", calibrated_prob * 100.0),
                "Calibration correction applied"
            );
        }

        // Log consensus estimate
        self.log_estimate(
            market,
            "bayesian",
            estimate.posterior,
            estimate.confidence,
            Some(calibrated_prob),
            Some(estimate.confidence),
            current_price,
        )
        .await;

        tracing::info!(
            market = %market.question,
            prior = format_args!("{:.1}%", current_price * 100.0),
            posterior = format_args!("{:.1}%", calibrated_prob * 100.0),
            lr = format_args!("{:.2}", estimate.combined_lr),
            conf = format_args!("{:.0}%", estimate.confidence * 100.0),
            "Bayesian assessment"
        );

        Ok((
            calibrated_prob,
            estimate.confidence,
            estimate.combined_lr,
            estimate.reasoning,
        ))
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
        let (mut news, source_counts) = self.news.fetch_all().await;
        let news_total = news.len();

        // Drop news older than 4 hours — stale news is already priced in
        let freshness_cutoff = chrono::Utc::now() - chrono::Duration::hours(4);
        news.retain(|item| {
            item.published
                .map(|p| p >= freshness_cutoff)
                .unwrap_or(true) // keep items without a date (can't tell age)
        });
        tracing::info!(
            fresh = news.len(),
            dropped = news_total - news.len(),
            "Filtered stale news (>4h old)"
        );

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
                news_matched: 0,
                llm_assessed: 0,
                source_counts,
            });
        }

        // Step 3: Match news to markets by semantic similarity (OpenAI embeddings)
        let matches = self.news.match_to_markets(&news, &eligible).await?;

        tracing::info!(
            matched = matches.len(),
            "Markets matched with relevant news"
        );

        let news_matched = matches.len();

        if matches.is_empty() {
            return Ok(ScanResult {
                signals: Vec::new(),
                rejections: Vec::new(),
                markets_scanned,
                news_total,
                news_new,
                news_matched: 0,
                llm_assessed: 0,
                source_counts,
            });
        }

        // Step 4: For top matches, check book depth and get price history
        let mut candidates: Vec<(NewsMatch, f64, String, Vec<PriceTick>, f64)> = Vec::new();

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

            candidates.push((
                nm.clone(),
                current_price,
                history_summary,
                history,
                book_depth,
            ));

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
                news_matched,
                llm_assessed: 0,
                source_counts,
            });
        }

        let has_model = self.xgb_model.is_some();
        tracing::info!(
            count = candidates.len(),
            xgb = has_model,
            "Assessing candidates{}",
            if has_model {
                " (XGBoost + LLM)"
            } else {
                " (LLM only)"
            }
        );

        // Step 5: Assess each candidate — XGBoost instant, LLM as fallback/comparison
        let mut signals = Vec::new();
        let mut rejections = Vec::new();

        for (i, (nm, current_price, history_summary, history, book_depth)) in
            candidates.iter().enumerate()
        {
            // Compute news features from matched articles
            let news_count = nm.news.len();
            let best_news_score = nm.relevance_score;
            let now = chrono::Utc::now();
            let avg_news_age_hours = if nm.news.is_empty() {
                0.0
            } else {
                let total_hours: f64 = nm
                    .news
                    .iter()
                    .map(|n| {
                        n.published
                            .map(|p| (now - p).num_minutes() as f64 / 60.0)
                            .unwrap_or(4.0)
                    })
                    .sum();
                total_hours / nm.news.len() as f64
            };

            // --- XGBoost prediction (instant, free) ---
            let xgb_estimate = self.xgb_model.as_ref().map(|model| {
                let features = MarketFeatures::from_market_and_news(
                    &nm.market,
                    *current_price,
                    history,
                    news_count,
                    best_news_score,
                    avg_news_age_hours,
                );
                let feature_vec = features.to_vec();
                let xgb_prob = model.predict_prob(&feature_vec);
                let xgb_conf = model.confidence(&feature_vec);
                (xgb_prob, xgb_conf)
            });

            if let Some((xgb_prob, xgb_conf)) = xgb_estimate {
                tracing::info!(
                    market = %nm.market.question,
                    xgb_prob = format_args!("{:.1}%", xgb_prob * 100.0),
                    xgb_conf = format_args!("{:.0}%", xgb_conf * 100.0),
                    price = format_args!("{:.1}%", current_price * 100.0),
                    news = news_count,
                    "XGBoost prediction"
                );
            }

            // --- Signal source: XGBoost primary, LLM fallback ---
            let (prob, confidence, combined_lr, reasoning, source) = if let Some((
                xgb_prob,
                xgb_conf,
            )) = xgb_estimate
            {
                // Use XGBoost as primary; construct a synthetic LR from the probability shift
                let lr = if *current_price > 0.01 && *current_price < 0.99 {
                    (xgb_prob / *current_price) / ((1.0 - xgb_prob) / (1.0 - *current_price))
                } else {
                    1.0
                };
                (
                    xgb_prob,
                    xgb_conf,
                    lr,
                    format!(
                        "XGBoost: {:.1}% (market {:.1}%, {} news)",
                        xgb_prob * 100.0,
                        current_price * 100.0,
                        news_count
                    ),
                    SignalSource::XgBoost,
                )
            } else {
                // No model — fall back to LLM for news assessment only
                if i > 0 {
                    tracing::info!("Waiting 21s for rate limit between candidates...");
                    tokio::time::sleep(Duration::from_secs(21)).await;
                }

                match self
                    .assess_news_impact(
                        &nm.market,
                        *current_price,
                        &nm.news,
                        history_summary,
                        past_bets_summary,
                    )
                    .await
                {
                    Ok(r) => (r.0, r.1, r.2, r.3, SignalSource::LlmConsensus),
                    Err(e) => {
                        tracing::warn!(market = %nm.market.market_id, err = %e, "LLM assessment failed");
                        continue;
                    }
                }
            };

            // Determine best side using Bayesian edge computation
            let estimate = BayesianEstimate {
                prior: *current_price,
                posterior: prob,
                combined_lr: 0.0, // not needed for edge calc
                confidence,
                reasoning: reasoning.clone(),
            };
            let (side, edge, bet_price, bet_prob) =
                match bayesian::compute_edge(&estimate, *current_price) {
                    Some((true, edge, price, prob)) => (BetSide::Yes, edge, price, prob),
                    Some((false, edge, price, prob)) => (BetSide::No, edge, price, prob),
                    None => {
                        let reason = format!(
                            "no edge (prob {:.0}% ~ price {:.0}%)",
                            prob * 100.0,
                            current_price * 100.0
                        );
                        tracing::info!(
                            market = %nm.market.question,
                            prob = format_args!("{:.1}%", prob * 100.0),
                            price = format_args!("{:.1}%", current_price * 100.0),
                            "No edge on either side"
                        );
                        rejections.push(RejectedSignal {
                            market_id: nm.market.market_id.clone(),
                            question: nm.market.question.clone(),
                            reason,
                            current_price: Some(*current_price),
                            estimated_prob: Some(prob),
                            edge: None,
                            confidence: Some(confidence),
                            combined_lr: Some(combined_lr),
                        });
                        continue;
                    }
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

            let reject = |reason: String| RejectedSignal {
                market_id: nm.market.market_id.clone(),
                question: nm.market.question.clone(),
                reason,
                current_price: Some(*current_price),
                estimated_prob: Some(prob),
                edge: Some(edge),
                confidence: Some(confidence),
                combined_lr: Some(combined_lr),
            };

            // Gate thresholds: trust XGBoost more (lower edge gate, lower confidence floor)
            let (min_edge, min_kelly, min_conf) = match source {
                SignalSource::XgBoost => (0.02, 0.003, 0.25),
                SignalSource::LlmConsensus => (0.03, 0.005, 0.30),
            };

            if effective_edge < min_edge {
                let reason = format!(
                    "edge {:.1}% < {:.0}%",
                    effective_edge * 100.0,
                    min_edge * 100.0
                );
                tracing::info!(
                    market = %nm.market.question,
                    eff_edge = format_args!("+{:.1}%", effective_edge * 100.0),
                    source = source.label(),
                    "REJECTED: effective edge below gate"
                );
                rejections.push(reject(reason));
                continue;
            }
            if kelly_size <= min_kelly {
                let reason = format!("kelly {:.3} < {:.3}", kelly_size, min_kelly);
                tracing::info!(
                    market = %nm.market.question,
                    kelly = format_args!("{:.3}", kelly_size),
                    "REJECTED: full Kelly too small"
                );
                rejections.push(reject(reason));
                continue;
            }
            if confidence < min_conf {
                let reason = format!("conf {:.0}% < {:.0}%", confidence * 100.0, min_conf * 100.0);
                tracing::info!(
                    market = %nm.market.question,
                    conf = format_args!("{:.0}%", confidence * 100.0),
                    "REJECTED: confidence below gate"
                );
                rejections.push(reject(reason));
                continue;
            }

            tracing::info!(
                market = %nm.market.question,
                side = %side,
                eff_edge = format_args!("+{:.1}%", effective_edge * 100.0),
                kelly = format_args!("{:.1}%", kelly_size * 100.0),
                conf = format_args!("{:.0}%", confidence * 100.0),
                source = source.label(),
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
                polymarket_url: nm.market.polymarket_url(),
                prior: *current_price,
                combined_lr,
                news_matched_count: nm.news.len(),
                source,
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

        let llm_assessed = candidates.len();

        tracing::info!(
            signals = signals.len(),
            rejections = rejections.len(),
            "Final news-driven signals"
        );
        Ok(ScanResult {
            signals,
            rejections,
            markets_scanned,
            news_total,
            news_new,
            news_matched,
            llm_assessed,
            source_counts,
        })
    }

    /// Get YES token IDs for all eligible markets (for websocket subscription).
    pub async fn eligible_token_ids(&self) -> Result<Vec<(String, String)>> {
        let markets = self.fetch_active_markets().await?;
        let pairs: Vec<(String, String)> = markets
            .into_iter()
            .filter(|m| m.volume_num >= self.cfg.min_volume)
            .filter(|m| self.expires_within_window(m.end_date.as_deref()))
            .filter(|m| {
                let price = Self::get_yes_price(m).unwrap_or(0.0);
                price > self.cfg.min_price && price < self.cfg.max_price
            })
            .filter_map(|m| {
                let token = m.yes_token_id()?;
                Some((token, m.market_id.clone()))
            })
            .collect();
        Ok(pairs)
    }

    /// Quickly assess a single market triggered by a websocket price move.
    /// Uses XGBoost only (instant, no LLM calls).
    pub async fn assess_alert(&self, market_id: &str, ws_price: f64) -> Result<Option<Signal>> {
        let model = match &self.xgb_model {
            Some(m) => m,
            None => return Ok(None),
        };

        // Fetch fresh market data
        let url = format!("{GAMMA_API}/markets/{market_id}");
        let resp = self.http.get(&url).send().await?;
        let market: GammaMarket = resp.json().await?;

        let current_price = Self::get_yes_price(&market).unwrap_or(ws_price);
        let token_id = match market.yes_token_id() {
            Some(t) => t,
            None => return Ok(None),
        };

        // Get price history for features
        let history = self
            .fetch_price_history(&token_id)
            .await
            .unwrap_or_default();

        let features = MarketFeatures::from_market_and_history(&market, current_price, &history);
        let feature_vec = features.to_vec();
        let xgb_prob = model.predict_prob(&feature_vec);
        let xgb_conf = model.confidence(&feature_vec);

        // Synthetic LR
        let lr = if current_price > 0.01 && current_price < 0.99 {
            (xgb_prob / current_price) / ((1.0 - xgb_prob) / (1.0 - current_price))
        } else {
            1.0
        };

        let estimate = BayesianEstimate {
            prior: current_price,
            posterior: xgb_prob,
            combined_lr: 0.0,
            confidence: xgb_conf,
            reasoning: String::new(),
        };

        let (side, edge, bet_price, bet_prob) =
            match bayesian::compute_edge(&estimate, current_price) {
                Some((true, edge, price, prob)) => (BetSide::Yes, edge, price, prob),
                Some((false, edge, price, prob)) => (BetSide::No, edge, price, prob),
                None => return Ok(None),
            };

        let kelly_size = fractional_kelly(bet_prob, bet_price, 1.0);
        let effective_edge = edge * xgb_conf;

        // Use XGBoost thresholds (trust the model)
        if effective_edge < 0.02 || kelly_size <= 0.003 || xgb_conf < 0.25 {
            return Ok(None);
        }

        let reasoning = format!(
            "XGBoost (WS trigger): {:.1}% (market {:.1}%)",
            xgb_prob * 100.0,
            current_price * 100.0
        );

        Ok(Some(Signal {
            market_id: market.market_id.clone(),
            question: market.question.clone(),
            side,
            current_price: bet_price,
            estimated_prob: bet_prob,
            confidence: xgb_conf,
            edge,
            kelly_size,
            reasoning,
            end_date: market.end_date.clone(),
            volume: market.volume_num,
            polymarket_url: market.polymarket_url(),
            prior: current_price,
            combined_lr: lr,
            news_matched_count: 0,
            source: SignalSource::XgBoost,
            context: BetContext {
                btc_price: 0.0,
                eth_price: 0.0,
                sol_price: 0.0,
                btc_24h_change: 0.0,
                btc_funding_rate: 0.0,
                btc_open_interest: 0.0,
                fear_greed: String::new(),
                book_depth: 0.0,
                news_headlines: Vec::new(),
            },
        }))
    }
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

/// Parsed LLM response — either a likelihood ratio (new) or raw probability (legacy).
enum LlmEstimate {
    LikelihoodRatio {
        lr: f64,
        confidence: f64,
        reasoning: String,
    },
    #[allow(dead_code)]
    RawProbability {
        probability: f64,
        confidence: f64,
        reasoning: String,
    },
}

fn parse_llm_response(response: &str) -> Result<LlmEstimate> {
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
        #[serde(default)]
        likelihood_ratio: Option<f64>,
        #[serde(default)]
        probability: Option<f64>,
        #[serde(default = "default_confidence")]
        confidence: f64,
        #[serde(default)]
        reasoning: String,
    }
    fn default_confidence() -> f64 {
        0.5
    }

    let est: Est = serde_json::from_str(json_str).context("failed to parse LLM response")?;

    if let Some(lr) = est.likelihood_ratio {
        Ok(LlmEstimate::LikelihoodRatio {
            lr: lr.clamp(0.1, 10.0),
            confidence: est.confidence.clamp(0.0, 1.0),
            reasoning: est.reasoning,
        })
    } else if let Some(prob) = est.probability {
        Ok(LlmEstimate::RawProbability {
            probability: prob.clamp(0.01, 0.99),
            confidence: est.confidence.clamp(0.0, 1.0),
            reasoning: est.reasoning,
        })
    } else {
        anyhow::bail!("LLM response missing both likelihood_ratio and probability")
    }
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

    // --- parse_llm_response ---

    #[test]
    fn test_parse_lr_json() {
        let input = r#"{"likelihood_ratio": 2.5, "confidence": 0.8, "reasoning": "strong signal"}"#;
        match parse_llm_response(input).unwrap() {
            LlmEstimate::LikelihoodRatio {
                lr,
                confidence,
                reasoning,
            } => {
                assert!((lr - 2.5).abs() < 1e-9);
                assert!((confidence - 0.8).abs() < 1e-9);
                assert_eq!(reasoning, "strong signal");
            }
            _ => panic!("expected LikelihoodRatio"),
        }
    }

    #[test]
    fn test_parse_lr_clamped() {
        let input = r#"{"likelihood_ratio": 50.0, "confidence": 0.9, "reasoning": "extreme"}"#;
        match parse_llm_response(input).unwrap() {
            LlmEstimate::LikelihoodRatio { lr, .. } => {
                assert!((lr - 10.0).abs() < 1e-9, "LR should clamp to 10.0");
            }
            _ => panic!("expected LikelihoodRatio"),
        }
    }

    #[test]
    fn test_parse_legacy_probability() {
        let input = r#"{"probability": 0.75, "confidence": 0.8, "reasoning": "strong signal"}"#;
        match parse_llm_response(input).unwrap() {
            LlmEstimate::RawProbability {
                probability,
                confidence,
                ..
            } => {
                assert!((probability - 0.75).abs() < 1e-9);
                assert!((confidence - 0.8).abs() < 1e-9);
            }
            _ => panic!("expected RawProbability"),
        }
    }

    #[test]
    fn test_parse_lr_preferred_over_probability() {
        // If both are present, LR takes precedence
        let input = r#"{"likelihood_ratio": 1.5, "probability": 0.75, "confidence": 0.8, "reasoning": "both"}"#;
        match parse_llm_response(input).unwrap() {
            LlmEstimate::LikelihoodRatio { lr, .. } => {
                assert!((lr - 1.5).abs() < 1e-9);
            }
            _ => panic!("expected LikelihoodRatio when both present"),
        }
    }

    #[test]
    fn test_parse_json_with_surrounding_text() {
        let input = "Here is my analysis:\n{\"likelihood_ratio\": 1.8, \"confidence\": 0.4, \"reasoning\": \"weak\"}\nThat's all.";
        match parse_llm_response(input).unwrap() {
            LlmEstimate::LikelihoodRatio { lr, confidence, .. } => {
                assert!((lr - 1.8).abs() < 1e-9);
                assert!((confidence - 0.4).abs() < 1e-9);
            }
            _ => panic!("expected LikelihoodRatio"),
        }
    }

    #[test]
    fn test_parse_clamps_extreme_values() {
        let input = r#"{"likelihood_ratio": 0.01, "confidence": -0.5, "reasoning": "extreme"}"#;
        match parse_llm_response(input).unwrap() {
            LlmEstimate::LikelihoodRatio { lr, confidence, .. } => {
                assert!((lr - 0.1).abs() < 1e-9, "LR should clamp to 0.1");
                assert!(confidence < 1e-9, "confidence should clamp to 0");
            }
            _ => panic!("expected LikelihoodRatio"),
        }
    }

    #[test]
    fn test_parse_missing_confidence_defaults() {
        let input = r#"{"likelihood_ratio": 1.2}"#;
        match parse_llm_response(input).unwrap() {
            LlmEstimate::LikelihoodRatio { confidence, .. } => {
                assert!((confidence - 0.5).abs() < 1e-9); // default_confidence
            }
            _ => panic!("expected LikelihoodRatio"),
        }
    }

    #[test]
    fn test_parse_missing_both_fails() {
        let input = r#"{"confidence": 0.8, "reasoning": "no estimate"}"#;
        assert!(parse_llm_response(input).is_err());
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
            polymarket_url: String::new(),
            prior: 0.50,
            combined_lr: 2.33,
            news_matched_count: 3,
            source: SignalSource::LlmConsensus,
            context: BetContext::default(),
        };
        let expected = 0.20 * 0.80 * 0.10;
        assert!((signal.score() - expected).abs() < 1e-9);
    }

    // --- parse edge cases ---

    #[test]
    fn test_parse_nested_json() {
        // LLM sometimes wraps in markdown code blocks
        let input = "```json\n{\"likelihood_ratio\": 1.5, \"confidence\": 0.6, \"reasoning\": \"test\"}\n```";
        match parse_llm_response(input).unwrap() {
            LlmEstimate::LikelihoodRatio { lr, .. } => {
                assert!((lr - 1.5).abs() < 1e-9);
            }
            _ => panic!("expected LikelihoodRatio"),
        }
    }

    #[test]
    fn test_parse_legacy_probability_boundaries() {
        // probability of exactly 0.0 should clamp to 0.01
        let input = r#"{"probability": 0.0, "confidence": 0.5, "reasoning": "no chance"}"#;
        match parse_llm_response(input).unwrap() {
            LlmEstimate::RawProbability { probability, .. } => {
                assert!((probability - 0.01).abs() < 1e-9);
            }
            _ => panic!("expected RawProbability"),
        }
    }
}
