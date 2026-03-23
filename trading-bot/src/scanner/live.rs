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
use crate::model::features::{MarketFeatures, OrderBookStats};
use crate::model::sidecar::SidecarClient;
use crate::pricing::kelly::fractional_kelly;
use crate::storage::portfolio::{BetContext, BetSide};
use crate::storage::postgres::RejectedSignal;

use super::news::{NewsAggregator, NewsItem, NewsMatch, dedup_news};

const GAMMA_API: &str = "https://gamma-api.polymarket.com";
const CLOB_API: &str = "https://clob.polymarket.com";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalSource {
    XgBoost,
    LlmConsensus,
    CopyTrade,
}

impl SignalSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::XgBoost => "xgboost",
            Self::LlmConsensus => "llm_consensus",
            Self::CopyTrade => "copy_trade",
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
    /// Days until market expiry (for terminal risk scaling)
    pub days_to_expiry: f64,
    /// Event slug grouping related markets (e.g. multi-outcome events)
    pub event_slug: Option<String>,
    /// Exact feature vector at signal time for online learning (ADR 004).
    pub features: Option<MarketFeatures>,
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
            SignalSource::CopyTrade => "\u{1f465} CopyTrade",
        };

        // Format reasoning block based on source
        let reasoning_block =
            if self.source == SignalSource::XgBoost || self.source == SignalSource::CopyTrade {
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

/// Result of the portfolio correlation check for a single bet candidate.
#[derive(Debug, Clone)]
pub struct CorrelationDecision {
    pub keep: bool,
    pub reason: String,
}

// RejectedSignal is imported from crate::storage::postgres (polymarket-common)

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

pub struct LiveScanner {
    http: Client,
    openai_client: openai::Client,
    news: NewsAggregator,
    pool: PgPool,
    calibration: RwLock<CalibrationCurve>,
    cfg: Arc<AppConfig>,
    sidecar: Option<SidecarClient>,
}

impl LiveScanner {
    pub async fn new(cfg: &Arc<AppConfig>, pool: PgPool) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client");
        let calibration = CalibrationCurve::load(&pool, cfg.calibration_min_samples).await?;

        // ML model sidecar (full ensemble) — required
        let sidecar = if !cfg.model_sidecar_url.is_empty() {
            let client = SidecarClient::new(&cfg.model_sidecar_url);
            if client.is_healthy().await {
                tracing::info!(url = %cfg.model_sidecar_url, "ML sidecar connected (full ensemble)");
            } else {
                tracing::warn!(url = %cfg.model_sidecar_url, "ML sidecar not healthy yet — predictions will fail until it's ready");
            }
            Some(client)
        } else {
            tracing::warn!("MODEL_SIDECAR_URL not set — model predictions disabled");
            None
        };

        Ok(Self {
            news: NewsAggregator::new(http.clone()),
            http,
            openai_client: openai::Client::from_env(),
            pool,
            calibration: RwLock::new(calibration),
            cfg: Arc::clone(cfg),
            sidecar,
        })
    }

    /// Reload calibration curve from DB (call periodically).
    pub async fn reload_calibration(&self) -> Result<()> {
        let curve = CalibrationCurve::load(&self.pool, self.cfg.calibration_min_samples).await?;
        *self.calibration.write().await = curve;
        Ok(())
    }

    /// Returns true if the ML sidecar is configured.
    pub fn has_model(&self) -> bool {
        self.sidecar.is_some()
    }

    /// Get model age in seconds from the sidecar health endpoint.
    pub async fn model_age_secs(&self) -> Option<f64> {
        let sidecar = self.sidecar.as_ref()?;
        sidecar.health().await?.model_age_secs
    }

    /// Trigger a warm-start retrain on the sidecar (fire-and-forget).
    /// Logs and swallows errors — warm-start failure must not block normal operation.
    pub async fn trigger_warmstart(&self) {
        let Some(sidecar) = &self.sidecar else { return };
        match sidecar.warmstart().await {
            Ok(()) => tracing::info!("Warm-start retrain triggered"),
            Err(e) => tracing::warn!(err = %e, "Warm-start retrain request failed"),
        }
    }

    /// Get prediction from the ML sidecar.
    async fn predict(&self, features: &MarketFeatures, market_price: f64) -> Option<(f64, f64)> {
        let sidecar = self.sidecar.as_ref()?;
        match sidecar.predict(features, market_price).await {
            Ok(pred) => {
                tracing::debug!(
                    market_price = format_args!("{:.1}%", market_price * 100.0),
                    ml_prob = format_args!("{:.1}%", pred.prob * 100.0),
                    ml_conf = format_args!("{:.0}%", pred.confidence * 100.0),
                    edge = format_args!("{:+.1}%", (pred.prob - market_price) * 100.0),
                    "Ensemble prediction"
                );
                Some((pred.prob, pred.confidence))
            }
            Err(e) => {
                tracing::warn!(err = %e, "Sidecar prediction failed");
                None
            }
        }
    }

    /// Batch prediction from the ML sidecar.
    async fn predict_batch(&self, items: &[(MarketFeatures, f64)]) -> Vec<Option<(f64, f64)>> {
        let Some(sidecar) = &self.sidecar else {
            return vec![None; items.len()];
        };
        match sidecar.predict_batch(items).await {
            Ok(preds) => {
                tracing::info!(
                    count = preds.len(),
                    avg_conf = format_args!(
                        "{:.0}%",
                        preds.iter().map(|p| p.confidence).sum::<f64>() / preds.len().max(1) as f64
                            * 100.0
                    ),
                    "Ensemble batch prediction complete"
                );
                preds
                    .into_iter()
                    .map(|p| Some((p.prob, p.confidence)))
                    .collect()
            }
            Err(e) => {
                tracing::warn!(err = %e, "Sidecar batch prediction failed");
                vec![None; items.len()]
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

    async fn fetch_order_book_stats(&self, token_id: &str) -> Result<OrderBookStats> {
        let url = format!("{CLOB_API}/book?token_id={token_id}");
        let resp: serde_json::Value = self
            .http
            .get(&url)
            .send()
            .await?
            .json()
            .await
            .context("failed to parse order book")?;

        let mut bid_volume = 0.0;
        let mut best_bid = 0.0_f64;
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
                bid_volume += price * size;
                best_bid = best_bid.max(price);
            }
        }

        let mut ask_volume = 0.0;
        let mut best_ask = 1.0_f64;
        if let Some(asks) = resp["asks"].as_array() {
            for ask in asks {
                let price = ask["price"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let size = ask["size"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                ask_volume += price * size;
                best_ask = best_ask.min(price);
            }
        }

        let total_volume = bid_volume + ask_volume;
        let order_imbalance = if total_volume > 0.0 {
            (bid_volume - ask_volume) / total_volume
        } else {
            0.0
        };
        let spread = (best_ask - best_bid).max(0.0);
        let depth = bid_volume + ask_volume;

        Ok(OrderBookStats {
            depth,
            order_imbalance,
            spread,
        })
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

    /// Fetch a single market using the query endpoint (includes `events` array).
    /// The `/markets/{id}` endpoint omits events; `/markets?id={id}` includes them.
    pub async fn fetch_market_with_events(&self, market_id: &str) -> Result<GammaMarket> {
        let url = format!("{GAMMA_API}/markets?id={market_id}");
        let resp = self.http.get(&url).send().await?;
        let text = resp.text().await?;
        let markets: Vec<GammaMarket> = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse market {market_id}"))?;
        markets
            .into_iter()
            .next()
            .with_context(|| format!("market {market_id} not found"))
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
        gm.yes_price()
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
        skip_event_slugs: &[String],
        past_bets_summary: &str,
        seen_headlines: &mut std::collections::HashSet<String>,
    ) -> Result<ScanResult> {
        // Step 1: Fetch all active markets
        let markets = self.fetch_active_markets().await?;
        let total = markets.len();
        tracing::info!(total, "Fetched active markets");

        let m = markets;
        let m: Vec<_> = m.into_iter().filter(|m| m.is_binary()).collect();
        let n_binary = m.len();
        let m: Vec<_> = m
            .into_iter()
            .filter(|m| m.volume_num >= self.cfg.min_volume)
            .collect();
        let n_volume = m.len();
        let m: Vec<_> = m
            .into_iter()
            .filter(|m| m.yes_token_id().is_some())
            .filter(|m| self.expires_within_window(m.end_date.as_deref()))
            .collect();
        let n_expiry = m.len();
        let m: Vec<_> = m
            .into_iter()
            .filter(|m| !skip_market_ids.contains(&m.market_id))
            .collect();
        let n_skip = m.len();
        let m: Vec<_> = m
            .into_iter()
            .filter(|m| match m.event_slug() {
                Some(slug) => !skip_event_slugs.iter().any(|s| s == slug),
                None => true,
            })
            .collect();
        let n_event = m.len();
        let eligible: Vec<GammaMarket> = m
            .into_iter()
            .filter(|m| {
                let price = Self::get_yes_price(m).unwrap_or(0.0);
                price > self.cfg.min_price && price < self.cfg.max_price
            })
            .collect();

        tracing::info!(
            total,
            binary = n_binary,
            volume = n_volume,
            expiry = n_expiry,
            skip_ids = n_skip,
            event_slug = n_event,
            eligible = eligible.len(),
            "Market filter breakdown (vol>${}, ≤{}d expiry)",
            self.cfg.min_volume,
            self.cfg.max_days_to_expiry,
        );

        let markets_scanned = eligible.len();

        if eligible.is_empty() {
            tracing::info!("No eligible markets, skipping scan");
            return Ok(ScanResult {
                signals: vec![],
                rejections: vec![],
                markets_scanned: 0,
                news_total: 0,
                news_new: 0,
                news_matched: 0,
                llm_assessed: 0,
                source_counts: vec![],
            });
        }

        // If we have a model, use MODEL-FIRST funnel.
        // Otherwise fall back to NEWS-FIRST funnel (LLM).
        if self.has_model() {
            self.scan_model_first(&eligible, skip_market_ids, seen_headlines, markets_scanned)
                .await
        } else {
            self.scan_news_first(
                &eligible,
                past_bets_summary,
                seen_headlines,
                markets_scanned,
            )
            .await
        }
    }

    /// MODEL-FIRST funnel: XGBoost on all markets → rank → top N → enrich with news.
    async fn scan_model_first(
        &self,
        eligible: &[GammaMarket],
        _skip_market_ids: &[String],
        _seen_headlines: &mut std::collections::HashSet<String>,
        markets_scanned: usize,
    ) -> Result<ScanResult> {
        let top_n = self.cfg.max_model_candidates;
        let use_sidecar = self.sidecar.is_some();

        tracing::info!(
            count = eligible.len(),
            source = if use_sidecar {
                "ensemble sidecar"
            } else {
                "local XGBoost"
            },
            "Running ML model on all eligible markets"
        );

        struct ModelCandidate {
            market: GammaMarket,
            current_price: f64,
            ml_prob: f64,
            ml_conf: f64,
            history: Vec<PriceTick>,
            book_stats: OrderBookStats,
            /// Model score for ranking: |edge| * confidence
            score: f64,
            /// Typed feature struct — used for re-prediction and online learning (ADR 004).
            features: MarketFeatures,
        }

        type MarketDataEntry = (GammaMarket, f64, Vec<PriceTick>, MarketFeatures);

        // Collect features and histories for all markets
        let mut market_data: Vec<MarketDataEntry> = Vec::new();
        for market in eligible {
            let current_price = Self::get_yes_price(market).unwrap_or(0.0);
            let token_id = match market.yes_token_id() {
                Some(t) => t,
                None => continue,
            };

            tokio::time::sleep(Duration::from_millis(50)).await;
            let history = self
                .fetch_price_history(&token_id)
                .await
                .unwrap_or_default();

            let features = MarketFeatures::from_market_and_history(market, current_price, &history);
            market_data.push((market.clone(), current_price, history, features));
        }

        // Batch predict via sidecar or per-item XGBoost
        let batch_items: Vec<(MarketFeatures, f64)> = market_data
            .iter()
            .map(|(_, price, _, features)| (features.clone(), *price))
            .collect();
        let predictions = self.predict_batch(&batch_items).await;

        let mut scored: Vec<ModelCandidate> = Vec::new();
        for ((market, current_price, history, features), pred) in
            market_data.into_iter().zip(predictions.into_iter())
        {
            let (ml_prob, ml_conf) = match pred {
                Some(p) => p,
                None => continue,
            };

            let edge = (ml_prob - current_price).abs();
            let score = edge * ml_conf;

            scored.push(ModelCandidate {
                market,
                current_price,
                ml_prob,
                ml_conf,
                history,
                book_stats: OrderBookStats::default(),
                score,
                features,
            });
        }

        // Rank by model score descending
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        tracing::info!(
            total = scored.len(),
            top_score = scored
                .first()
                .map(|c| format!("{:.3}", c.score))
                .unwrap_or_default(),
            "XGBoost scored all markets"
        );

        // Step 3: Take top N, check liquidity
        let mut candidates: Vec<ModelCandidate> = Vec::new();
        for mut c in scored.into_iter().take(top_n * 2) {
            let token_id = match c.market.yes_token_id() {
                Some(t) => t,
                None => continue,
            };

            tokio::time::sleep(Duration::from_millis(50)).await;
            let book_stats = self
                .fetch_order_book_stats(&token_id)
                .await
                .unwrap_or_default();
            if book_stats.depth < self.cfg.min_book_depth {
                tracing::debug!(
                    market = %c.market.question,
                    depth = format_args!("${:.0}", book_stats.depth),
                    "Skipping — thin book"
                );
                continue;
            }
            c.book_stats = book_stats;
            candidates.push(c);
            if candidates.len() >= top_n {
                break;
            }
        }

        if candidates.is_empty() {
            tracing::info!("No model candidates passed liquidity filter");
            return Ok(ScanResult {
                signals: Vec::new(),
                rejections: Vec::new(),
                markets_scanned,
                news_total: 0,
                news_new: 0,
                news_matched: 0,
                llm_assessed: 0,
                source_counts: Vec::new(),
            });
        }

        tracing::info!(
            count = candidates.len(),
            "Top model candidates passed liquidity"
        );

        // Step 4: Optionally fetch news and match to top candidates (Telegram context only).
        // News has no effect on model predictions — disable via NEWS_ENABLED=false to save
        // RSS + OpenAI embedding costs when running with a model sidecar.
        let (news_total, news_new, news_matched, news_by_market) = if self.cfg.news_enabled {
            let (mut news, _source_counts) = self.news.fetch_all().await;
            let news_total = news.len();

            let freshness_cutoff = chrono::Utc::now() - chrono::Duration::hours(4);
            news.retain(|item| {
                item.published
                    .map(|p| p >= freshness_cutoff)
                    .unwrap_or(true)
            });
            dedup_news(&mut news);
            let news_new = news.len();

            let candidate_markets: Vec<GammaMarket> =
                candidates.iter().map(|c| c.market.clone()).collect();
            let news_matches = if !news.is_empty() {
                self.news
                    .match_to_markets(&news, &candidate_markets)
                    .await
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let news_matched = news_matches.len();
            let mut by_market: std::collections::HashMap<String, NewsMatch> =
                std::collections::HashMap::new();
            for nm in news_matches {
                by_market.insert(nm.market.market_id.clone(), nm);
            }
            tracing::info!(
                news_total,
                news_new,
                news_matched,
                "News fetched and matched"
            );
            (news_total, news_new, news_matched, by_market)
        } else {
            tracing::debug!("News disabled (NEWS_ENABLED=false) — skipping fetch");
            (0, 0, 0, std::collections::HashMap::new())
        };
        let source_counts: Vec<(String, usize)> = Vec::new();
        let news_by_market = &news_by_market;

        // Step 5: Build signals from model candidates + optional news boost
        let mut signals = Vec::new();
        let mut rejections = Vec::new();

        for c in &candidates {
            let now = chrono::Utc::now();

            // Check if we have matched news for this market
            let (news_count, best_news_score, avg_news_age_hours, matched_news) =
                if let Some(nm) = news_by_market.get(&c.market.market_id) {
                    let age = if nm.news.is_empty() {
                        0.0
                    } else {
                        let total: f64 = nm
                            .news
                            .iter()
                            .map(|n| {
                                n.published
                                    .map(|p| (now - p).num_minutes() as f64 / 60.0)
                                    .unwrap_or(4.0)
                            })
                            .sum();
                        total / nm.news.len() as f64
                    };
                    (nm.news.len(), nm.relevance_score, age, nm.news.clone())
                } else {
                    (0, 0.0, 0.0, Vec::new())
                };

            // Re-run model with news + order book features
            let (ml_prob, ml_conf, signal_features) = if news_count > 0 {
                let features = MarketFeatures::from_market_and_news(
                    &c.market,
                    c.current_price,
                    &c.history,
                    news_count,
                    best_news_score,
                    avg_news_age_hours,
                );
                match self.predict(&features, c.current_price).await {
                    Some(p) => (p.0, p.1, features),
                    None => (c.ml_prob, c.ml_conf, c.features.clone()),
                }
            } else {
                let features =
                    MarketFeatures::from_market_and_history(&c.market, c.current_price, &c.history);
                match self.predict(&features, c.current_price).await {
                    Some(p) => (p.0, p.1, features),
                    None => (c.ml_prob, c.ml_conf, c.features.clone()),
                }
            };

            let lr = if c.current_price > 0.01 && c.current_price < 0.99 {
                (ml_prob / c.current_price) / ((1.0 - ml_prob) / (1.0 - c.current_price))
            } else {
                1.0
            };

            let reasoning = if news_count > 0 {
                format!(
                    "ML: {:.1}% (market {:.1}%, {} news matched)",
                    ml_prob * 100.0,
                    c.current_price * 100.0,
                    news_count,
                )
            } else {
                format!(
                    "ML: {:.1}% (market {:.1}%)",
                    ml_prob * 100.0,
                    c.current_price * 100.0,
                )
            };

            tracing::info!(
                market = %c.market.question,
                ml_prob = format_args!("{:.1}%", ml_prob * 100.0),
                ml_conf = format_args!("{:.0}%", ml_conf * 100.0),
                price = format_args!("{:.1}%", c.current_price * 100.0),
                news = news_count,
                "Model candidate"
            );

            // Bayesian update: anchor model prediction to market price.
            // Market price is the prior (encodes all public info),
            // model's LR is dampened by its confidence before updating.
            let dampened_lr = bayesian::dampen_lr(lr, ml_conf);
            let prior_odds = bayesian::prob_to_odds(c.current_price);
            let posterior = bayesian::odds_to_prob(prior_odds * dampened_lr);

            let estimate = BayesianEstimate {
                posterior,
                combined_lr: dampened_lr,
                confidence: ml_conf,
                reasoning: reasoning.clone(),
            };
            let (side, edge, bet_price, bet_prob) =
                match bayesian::compute_edge(&estimate, c.current_price) {
                    Some((true, e, p, pr)) => (BetSide::Yes, e, p, pr),
                    Some((false, e, p, pr)) => (BetSide::No, e, p, pr),
                    None => {
                        rejections.push(RejectedSignal {
                            market_id: c.market.market_id.clone(),
                            question: c.market.question.clone(),
                            reason: format!(
                                "no edge (prob {:.0}% ~ price {:.0}%)",
                                ml_prob * 100.0,
                                c.current_price * 100.0
                            ),
                            current_price: Some(c.current_price),
                            estimated_prob: Some(ml_prob),
                            edge: None,
                            confidence: Some(ml_conf),
                            combined_lr: Some(lr),
                        });
                        continue;
                    }
                };

            let kelly_size = fractional_kelly(bet_prob, bet_price, 1.0);
            let effective_edge = edge * ml_conf;

            let reject = |reason: String| RejectedSignal {
                market_id: c.market.market_id.clone(),
                question: c.market.question.clone(),
                reason,
                current_price: Some(c.current_price),
                estimated_prob: Some(ml_prob),
                edge: Some(edge),
                confidence: Some(ml_conf),
                combined_lr: Some(lr),
            };

            // Gate thresholds
            if effective_edge < 0.02 {
                rejections.push(reject(format!(
                    "edge {:.1}% < 2.0%",
                    effective_edge * 100.0,
                )));
                continue;
            }
            if kelly_size <= 0.003 {
                rejections.push(reject(format!("kelly {:.3} < 0.003", kelly_size)));
                continue;
            }
            if ml_conf < 0.25 {
                rejections.push(reject(format!("conf {:.0}% < 25%", ml_conf * 100.0)));
                continue;
            }

            tracing::info!(
                market = %c.market.question,
                side = %side,
                eff_edge = format_args!("+{:.1}%", effective_edge * 100.0),
                kelly = format_args!("{:.1}%", kelly_size * 100.0),
                conf = format_args!("{:.0}%", ml_conf * 100.0),
                news = news_count,
                "SIGNAL — passed model filters"
            );

            let news_headlines: Vec<String> =
                matched_news.iter().map(|n| n.title.clone()).collect();
            signals.push(Signal {
                market_id: c.market.market_id.clone(),
                question: c.market.question.clone(),
                side,
                current_price: bet_price,
                estimated_prob: bet_prob,
                confidence: ml_conf,
                edge,
                kelly_size,
                reasoning,
                end_date: c.market.end_date.clone(),
                volume: c.market.volume_num,
                polymarket_url: c.market.polymarket_url(),
                prior: c.current_price,
                combined_lr: lr,
                news_matched_count: news_count,
                source: SignalSource::XgBoost,
                days_to_expiry: parse_days_to_expiry(&c.market.end_date),
                event_slug: c.market.event_slug().map(String::from),
                context: BetContext {
                    btc_price: 0.0,
                    eth_price: 0.0,
                    sol_price: 0.0,
                    btc_24h_change: 0.0,
                    btc_funding_rate: 0.0,
                    btc_open_interest: 0.0,
                    fear_greed: String::new(),
                    book_depth: c.book_stats.depth,
                    news_headlines,
                },
                features: Some(signal_features),
            });
        }

        signals.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap());

        tracing::info!(
            signals = signals.len(),
            rejections = rejections.len(),
            "Final model-first signals"
        );
        Ok(ScanResult {
            signals,
            rejections,
            markets_scanned,
            news_total,
            news_new,
            news_matched,
            llm_assessed: candidates.len(),
            source_counts,
        })
    }

    /// NEWS-FIRST funnel (fallback when no XGBoost model): news → match → LLM → bet.
    async fn scan_news_first(
        &self,
        eligible: &[GammaMarket],
        past_bets_summary: &str,
        seen_headlines: &mut std::collections::HashSet<String>,
        markets_scanned: usize,
    ) -> Result<ScanResult> {
        // Fetch news
        let (mut news, source_counts) = self.news.fetch_all().await;
        let news_total = news.len();

        let freshness_cutoff = chrono::Utc::now() - chrono::Duration::hours(4);
        news.retain(|item| {
            item.published
                .map(|p| p >= freshness_cutoff)
                .unwrap_or(true)
        });

        // Dedup seen headlines
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

        // Match news to markets
        let matches = self.news.match_to_markets(&news, eligible).await?;
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

        // Fetch depth + history for top matches
        let mut candidates: Vec<(NewsMatch, f64, String, Vec<PriceTick>, OrderBookStats)> =
            Vec::new();
        for nm in matches.iter().take(self.cfg.max_llm_candidates * 2) {
            let token_id = nm.market.yes_token_id().unwrap();
            let current_price = Self::get_yes_price(&nm.market).unwrap_or(0.0);

            tokio::time::sleep(Duration::from_millis(100)).await;
            let book_stats = self
                .fetch_order_book_stats(&token_id)
                .await
                .unwrap_or_default();
            if book_stats.depth < self.cfg.min_book_depth {
                continue;
            }

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
                book_stats,
            ));
            if candidates.len() >= self.cfg.max_llm_candidates {
                break;
            }
        }

        if candidates.is_empty() {
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

        tracing::info!(count = candidates.len(), "Assessing candidates (LLM only)");

        // LLM assessment for each candidate
        let mut signals = Vec::new();
        let mut rejections = Vec::new();

        for (i, (nm, current_price, history_summary, history, book_stats)) in
            candidates.iter().enumerate()
        {
            if i > 0 {
                tracing::info!("Waiting 21s for rate limit...");
                tokio::time::sleep(Duration::from_secs(21)).await;
            }

            let (prob, confidence, combined_lr, reasoning) = match self
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
                    tracing::warn!(market = %nm.market.market_id, err = %e, "LLM failed");
                    continue;
                }
            };

            let estimate = BayesianEstimate {
                posterior: prob,
                combined_lr: 0.0,
                confidence,
                reasoning: reasoning.clone(),
            };
            let (side, edge, bet_price, bet_prob) =
                match bayesian::compute_edge(&estimate, *current_price) {
                    Some((true, e, p, pr)) => (BetSide::Yes, e, p, pr),
                    Some((false, e, p, pr)) => (BetSide::No, e, p, pr),
                    None => {
                        rejections.push(RejectedSignal {
                            market_id: nm.market.market_id.clone(),
                            question: nm.market.question.clone(),
                            reason: format!(
                                "no edge (prob {:.0}% ~ price {:.0}%)",
                                prob * 100.0,
                                current_price * 100.0
                            ),
                            current_price: Some(*current_price),
                            estimated_prob: Some(prob),
                            edge: None,
                            confidence: Some(confidence),
                            combined_lr: Some(combined_lr),
                        });
                        continue;
                    }
                };

            let kelly_size = fractional_kelly(bet_prob, bet_price, 1.0);
            let effective_edge = edge * confidence;

            // LLM gate thresholds (stricter)
            if effective_edge < 0.03 || kelly_size <= 0.005 || confidence < 0.30 {
                rejections.push(RejectedSignal {
                    market_id: nm.market.market_id.clone(),
                    question: nm.market.question.clone(),
                    reason: format!(
                        "LLM gate: edge={:.1}% kelly={:.3} conf={:.0}%",
                        effective_edge * 100.0,
                        kelly_size,
                        confidence * 100.0,
                    ),
                    current_price: Some(*current_price),
                    estimated_prob: Some(prob),
                    edge: Some(edge),
                    confidence: Some(confidence),
                    combined_lr: Some(combined_lr),
                });
                continue;
            }

            let news_headlines: Vec<String> = nm.news.iter().map(|n| n.title.clone()).collect();
            let signal_features =
                MarketFeatures::from_market_and_history(&nm.market, *current_price, history);
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
                source: SignalSource::LlmConsensus,
                days_to_expiry: parse_days_to_expiry(&nm.market.end_date),
                event_slug: nm.market.event_slug().map(String::from),
                context: BetContext {
                    btc_price: 0.0,
                    eth_price: 0.0,
                    sol_price: 0.0,
                    btc_24h_change: 0.0,
                    btc_funding_rate: 0.0,
                    btc_open_interest: 0.0,
                    fear_greed: String::new(),
                    book_depth: book_stats.depth,
                    news_headlines,
                },
                features: Some(signal_features),
            });
        }

        signals.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap());

        Ok(ScanResult {
            signals,
            rejections,
            markets_scanned,
            news_total,
            news_new,
            news_matched,
            llm_assessed: candidates.len(),
            source_counts,
        })
    }

    /// Get YES token IDs for all eligible markets (for websocket subscription).
    pub async fn eligible_token_ids(&self) -> Result<Vec<(String, String)>> {
        let markets = self.fetch_active_markets().await?;
        let pairs: Vec<(String, String)> = markets
            .into_iter()
            .filter(|m| m.is_binary())
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
    /// Uses ML model (sidecar or local XGBoost fallback).
    pub async fn assess_alert(&self, market_id: &str, ws_price: f64) -> Result<Option<Signal>> {
        if !self.has_model() {
            return Ok(None);
        }

        // Fetch fresh market data (query endpoint includes events for event_slug)
        let market = self.fetch_market_with_events(market_id).await?;

        let current_price = Self::get_yes_price(&market).unwrap_or(ws_price);
        let token_id = match market.yes_token_id() {
            Some(t) => t,
            None => return Ok(None),
        };

        let history = self
            .fetch_price_history(&token_id)
            .await
            .unwrap_or_default();

        let features = MarketFeatures::from_market_and_history(&market, current_price, &history);
        let (ml_prob, ml_conf) = match self.predict(&features, current_price).await {
            Some(p) => p,
            None => return Ok(None),
        };

        let lr = if current_price > 0.01 && current_price < 0.99 {
            (ml_prob / current_price) / ((1.0 - ml_prob) / (1.0 - current_price))
        } else {
            1.0
        };

        let estimate = BayesianEstimate {
            posterior: ml_prob,
            combined_lr: 0.0,
            confidence: ml_conf,
            reasoning: String::new(),
        };

        let (side, edge, bet_price, bet_prob) =
            match bayesian::compute_edge(&estimate, current_price) {
                Some((true, edge, price, prob)) => (BetSide::Yes, edge, price, prob),
                Some((false, edge, price, prob)) => (BetSide::No, edge, price, prob),
                None => return Ok(None),
            };

        let kelly_size = fractional_kelly(bet_prob, bet_price, 1.0);
        let effective_edge = edge * ml_conf;

        // WS-triggered: stricter thresholds
        if effective_edge < 0.05 || kelly_size <= 0.01 || ml_conf < 0.40 {
            return Ok(None);
        }

        let reasoning = format!(
            "ML (WS trigger): {:.1}% (market {:.1}%)",
            ml_prob * 100.0,
            current_price * 100.0
        );

        Ok(Some(Signal {
            market_id: market.market_id.clone(),
            question: market.question.clone(),
            side,
            current_price: bet_price,
            estimated_prob: bet_prob,
            confidence: ml_conf,
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
            days_to_expiry: parse_days_to_expiry(&market.end_date),
            event_slug: market.event_slug().map(String::from),
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
            features: Some(features),
        }))
    }

    /// Check all bet candidates against existing open bets in a single LLM call.
    ///
    /// Sends only question + side for each candidate and open bet — no amounts.
    /// Returns one `CorrelationDecision` per candidate.
    /// Fails open: on any error every candidate is returned with `keep = true`.
    pub async fn check_portfolio_correlation(
        &self,
        candidates: &[(String, String, BetSide)],
        open_bets: &[crate::storage::portfolio::Bet],
        blocked: &[(String, BetSide)],
    ) -> Result<Vec<CorrelationDecision>> {
        let mut open: Vec<(String, BetSide)> = open_bets
            .iter()
            .map(|b| (b.question.clone(), b.side.clone()))
            .collect();
        open.extend(blocked.iter().cloned());
        run_correlation_check(&self.openai_client, &self.cfg.llm_model, candidates, &open).await
    }
}

/// Core logic for the portfolio correlation check, decoupled from `LiveScanner`
/// so it can be called directly in tests.
async fn run_correlation_check(
    openai_client: &openai::Client,
    model: &str,
    candidates: &[(String, String, BetSide)],
    open_bets: &[(String, BetSide)],
) -> Result<Vec<CorrelationDecision>> {
    if candidates.is_empty() {
        return Ok(vec![]);
    }

    let candidate_ids: Vec<char> = (b'A'..=b'Z')
        .take(candidates.len())
        .map(|c| c as char)
        .collect();

    let open_section = if open_bets.is_empty() {
        "None".to_string()
    } else {
        open_bets
            .iter()
            .enumerate()
            .map(|(i, (q, s))| format!("{}. {} — \"{}\"", i + 1, s, q))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let candidate_section = candidates
        .iter()
        .enumerate()
        .map(|(i, (_, q, s))| format!("{}. {} — \"{}\"", candidate_ids[i], s, q))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "EXISTING OPEN BETS:\n{open_section}\n\n\
         NEW BET CANDIDATES:\n{candidate_section}\n\n\
         Respond with ONLY a JSON array, one object per candidate:\n\
         [{{\"id\":\"A\",\"keep\":true,\"reason\":\"...\"}}]\n\
         - keep: false only if the candidate is logically correlated with, mutually \
           exclusive with, or essentially the same event as an existing open bet\n\
         - keep: true if the candidate is genuinely independent\n\
         - reason: one short sentence"
    );

    let agent = openai_client
        .agent(model)
        .preamble(
            "You are a portfolio overlap detector for a prediction market trading bot.\n\
             REJECT a candidate only for clear overlaps:\n\
             - Threshold ladders: e.g. NO on oil hitting $100 while already NO on oil hitting $105 \
               (if $100 hits so does $105 — positions are entangled)\n\
             - Partition buckets: e.g. YES on tweet range 260-279 while already YES on 280-299 \
               for the same period (at most one resolves YES)\n\
             - Same event different framing: essentially identical question under a different title\n\
             KEEP everything else. Be conservative — only reject clear logical overlaps.",
        )
        .temperature(0.0)
        .build();

    let response = agent
        .chat(&prompt, vec![])
        .await
        .map_err(|e| anyhow::anyhow!("Correlation check LLM call failed: {e}"))?;

    tracing::info!(response = %response, "Correlation check raw response");

    let json_str = match extract_json_array(&response) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(err = %e, "Failed to extract JSON array from correlation response, keeping all");
            return Ok(candidates
                .iter()
                .map(|_| CorrelationDecision {
                    keep: true,
                    reason: "Parse error — defaulting to keep".to_string(),
                })
                .collect());
        }
    };

    let items: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(err = %e, "Failed to parse correlation JSON, keeping all");
            return Ok(candidates
                .iter()
                .map(|_| CorrelationDecision {
                    keep: true,
                    reason: "Parse error — defaulting to keep".to_string(),
                })
                .collect());
        }
    };

    Ok(parse_correlation_decisions(&items, &candidate_ids))
}

/// Map parsed JSON items back to `CorrelationDecision`s using letter IDs (A, B, C...).
/// Missing entries default to keep=true.
fn parse_correlation_decisions(
    items: &[serde_json::Value],
    candidate_ids: &[char],
) -> Vec<CorrelationDecision> {
    candidate_ids
        .iter()
        .map(|id| {
            let id_str = id.to_string();
            let item = items.iter().find(|v| v["id"].as_str() == Some(&id_str));
            match item {
                Some(v) => CorrelationDecision {
                    keep: v["keep"].as_bool().unwrap_or(true),
                    reason: v["reason"]
                        .as_str()
                        .unwrap_or("no reason provided")
                        .to_string(),
                },
                None => CorrelationDecision {
                    keep: true,
                    reason: "No decision returned — defaulting to keep".to_string(),
                },
            }
        })
        .collect()
}

fn extract_json_array(response: &str) -> Result<String> {
    if let Some(start) = response.find('[')
        && let Some(end) = response[start..].rfind(']')
    {
        return Ok(response[start..=start + end].to_string());
    }
    anyhow::bail!("No JSON array found in correlation check response")
}

pub fn parse_days_to_expiry(end_date: &Option<String>) -> f64 {
    end_date
        .as_ref()
        .and_then(|d| chrono::DateTime::parse_from_rfc3339(d).ok())
        .map(|end| {
            let remaining = end.timestamp() - chrono::Utc::now().timestamp();
            (remaining as f64 / 86400.0).max(0.0)
        })
        .unwrap_or(30.0)
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
            days_to_expiry: 7.0,
            event_slug: None,
            context: BetContext::default(),
            features: None,
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

    // --- extract_json_array ---

    #[test]
    fn test_extract_json_array_clean() {
        let input = r#"[{"id":"A","keep":true,"reason":"independent"}]"#;
        assert_eq!(extract_json_array(input).unwrap(), input);
    }

    #[test]
    fn test_extract_json_array_with_surrounding_text() {
        let input = "Here is my analysis:\n[{\"id\":\"A\",\"keep\":false,\"reason\":\"correlated\"}]\nDone.";
        let extracted = extract_json_array(input).unwrap();
        assert!(extracted.starts_with('['));
        assert!(extracted.ends_with(']'));
        assert!(extracted.contains("correlated"));
    }

    #[test]
    fn test_extract_json_array_in_markdown_block() {
        let input = "```json\n[{\"id\":\"A\",\"keep\":true,\"reason\":\"ok\"}]\n```";
        let extracted = extract_json_array(input).unwrap();
        assert!(extracted.starts_with('['));
        assert!(extracted.contains("ok"));
    }

    #[test]
    fn test_extract_json_array_missing_fails() {
        assert!(extract_json_array("no array here {}").is_err());
        assert!(extract_json_array("").is_err());
    }

    // --- parse_correlation_decisions ---

    #[test]
    fn test_parse_correlation_all_keep() {
        let json: Vec<serde_json::Value> = serde_json::from_str(
            r#"[
                {"id":"A","keep":true,"reason":"independent event"},
                {"id":"B","keep":true,"reason":"different topic"}
            ]"#,
        )
        .unwrap();
        let ids = vec!['A', 'B'];
        let decisions = parse_correlation_decisions(&json, &ids);
        assert_eq!(decisions.len(), 2);
        assert!(decisions[0].keep);
        assert_eq!(decisions[0].reason, "independent event");
        assert!(decisions[1].keep);
    }

    #[test]
    fn test_parse_correlation_reject_one() {
        let json: Vec<serde_json::Value> = serde_json::from_str(
            r#"[
                {"id":"A","keep":false,"reason":"tweet bucket overlap with open bet"},
                {"id":"B","keep":true,"reason":"unrelated market"}
            ]"#,
        )
        .unwrap();
        let ids = vec!['A', 'B'];
        let decisions = parse_correlation_decisions(&json, &ids);
        assert!(!decisions[0].keep);
        assert_eq!(decisions[0].reason, "tweet bucket overlap with open bet");
        assert!(decisions[1].keep);
    }

    #[test]
    fn test_parse_correlation_missing_entry_defaults_keep() {
        // LLM only returns decision for A, skips B
        let json: Vec<serde_json::Value> =
            serde_json::from_str(r#"[{"id":"A","keep":false,"reason":"correlated"}]"#).unwrap();
        let ids = vec!['A', 'B'];
        let decisions = parse_correlation_decisions(&json, &ids);
        assert!(!decisions[0].keep);
        assert!(decisions[1].keep, "missing entry should default to keep");
        assert_eq!(
            decisions[1].reason,
            "No decision returned — defaulting to keep"
        );
    }

    #[test]
    fn test_parse_correlation_missing_keep_field_defaults_true() {
        // keep field omitted — should default to true (fail-open)
        let json: Vec<serde_json::Value> =
            serde_json::from_str(r#"[{"id":"A","reason":"something"}]"#).unwrap();
        let ids = vec!['A'];
        let decisions = parse_correlation_decisions(&json, &ids);
        assert!(decisions[0].keep);
    }

    #[test]
    fn test_parse_correlation_empty_candidates() {
        let json: Vec<serde_json::Value> = vec![];
        let ids: Vec<char> = vec![];
        let decisions = parse_correlation_decisions(&json, &ids);
        assert!(decisions.is_empty());
    }

    // --- run_correlation_check (real OpenAI) ---

    /// Integration test — requires OPENAI_API_KEY in environment.
    /// Run with: cargo test test_correlation_check_real_llm -- --ignored
    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn test_correlation_check_real_llm() {
        let client = openai::Client::from_env();

        // Scenario: two partition buckets (should be rejected) + one independent (should be kept)
        let candidates = vec![
            (
                "market-a".to_string(),
                "Will Elon Musk post 280-299 tweets from March 10 to March 17?".to_string(),
                BetSide::Yes,
            ),
            (
                "market-b".to_string(),
                "Will Crude Oil CL hit HIGH $100 by end of March?".to_string(),
                BetSide::No,
            ),
            (
                "market-c".to_string(),
                "Will France win the 2026 FIFA World Cup?".to_string(),
                BetSide::Yes,
            ),
        ];

        let open_bets = vec![
            (
                "Will Elon Musk post 260-279 tweets from March 10 to March 17?".to_string(),
                BetSide::Yes,
            ),
            (
                "Will Crude Oil CL hit HIGH $105 by end of March?".to_string(),
                BetSide::No,
            ),
        ];

        let decisions = run_correlation_check(&client, "gpt-4o", &candidates, &open_bets)
            .await
            .expect("LLM call failed");

        assert_eq!(decisions.len(), 3);

        // A: tweet bucket 280-299 vs open bet on 260-279 — same period, partition → reject
        assert!(
            !decisions[0].keep,
            "tweet bucket overlap should be rejected, got: {}",
            decisions[0].reason
        );

        // B: oil $100 NO vs open bet oil $105 NO — threshold ladder → reject
        assert!(
            !decisions[1].keep,
            "oil threshold ladder should be rejected, got: {}",
            decisions[1].reason
        );

        // C: FIFA World Cup — unrelated → keep
        assert!(
            decisions[2].keep,
            "independent market should be kept, got: {}",
            decisions[2].reason
        );
    }

    #[test]
    fn test_parse_correlation_out_of_order_ids() {
        // LLM returns B before A — should still map correctly
        let json: Vec<serde_json::Value> = serde_json::from_str(
            r#"[
                {"id":"B","keep":false,"reason":"threshold ladder"},
                {"id":"A","keep":true,"reason":"independent"}
            ]"#,
        )
        .unwrap();
        let ids = vec!['A', 'B'];
        let decisions = parse_correlation_decisions(&json, &ids);
        assert!(decisions[0].keep, "A should be keep=true");
        assert!(!decisions[1].keep, "B should be keep=false");
        assert_eq!(decisions[1].reason, "threshold ladder");
    }
}
