use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use reqwest::Client;
use rig::completion::Chat;
use rig::providers::openai;
use std::time::Duration;

use crate::data::models::{GammaMarket, PriceTick};
use crate::pricing::kelly::fractional_kelly;

const GAMMA_API: &str = "https://gamma-api.polymarket.com";
const CLOB_API: &str = "https://clob.polymarket.com";

/// Minimum expected edge (edge * confidence) to emit a signal.
const MIN_EDGE_PCT: f64 = 0.10;
/// Minimum volume to consider a market liquid enough.
const MIN_VOLUME: f64 = 5000.0;
/// Kelly fraction — conservative quarter-Kelly.
const KELLY_FRACTION: f64 = 0.25;
/// Maximum days until resolution.
const MAX_DAYS_TO_EXPIRY: i64 = 7;

#[derive(Debug, Clone)]
pub struct Signal {
    pub market_id: String,
    pub question: String,
    pub current_price: f64,
    pub estimated_prob: f64,
    pub confidence: f64,
    pub edge: f64,
    pub kelly_size: f64,
    pub reasoning: String,
    pub end_date: Option<String>,
    pub volume: f64,
}

impl Signal {
    pub fn to_telegram_message(&self) -> String {
        format!(
            "🟢 *YES Signal*\n\n\
             📋 *{question}*\n\n\
             💰 Current price: `{price:.1}¢`\n\
             🎯 Our estimate: `{est:.1}%`\n\
             📊 Edge: `+{edge:.1}%`\n\
             🔒 Confidence: `{conf:.0}%`\n\
             📐 Kelly size: `{kelly:.1}%` of bankroll\n\
             💧 Volume: `${vol:.0}`\n\
             ⏰ Expires: {end}\n\n\
             💡 _{reasoning}_",
            question = self.question,
            price = self.current_price * 100.0,
            est = self.estimated_prob * 100.0,
            edge = self.edge * 100.0,
            conf = self.confidence * 100.0,
            kelly = self.kelly_size * 100.0,
            vol = self.volume,
            end = self.end_date.as_deref().unwrap_or("N/A"),
            reasoning = self.reasoning,
        )
    }

    pub fn score(&self) -> f64 {
        self.edge * self.confidence * self.kelly_size
    }
}

/// Live crypto prices fetched from CoinGecko.
#[derive(Debug)]
struct CryptoContext {
    btc_price: f64,
    eth_price: f64,
    sol_price: f64,
    btc_24h_change: f64,
    eth_24h_change: f64,
    sol_24h_change: f64,
}

impl CryptoContext {
    fn summary(&self) -> String {
        format!(
            "LIVE CRYPTO PRICES:\n\
             BTC: ${:.0} ({:+.1}% 24h)\n\
             ETH: ${:.0} ({:+.1}% 24h)\n\
             SOL: ${:.1} ({:+.1}% 24h)",
            self.btc_price,
            self.btc_24h_change,
            self.eth_price,
            self.eth_24h_change,
            self.sol_price,
            self.sol_24h_change,
        )
    }
}

pub struct LiveScanner {
    http: Client,
    llm: rig::agent::Agent<openai::CompletionModel>,
}

impl LiveScanner {
    pub fn new() -> Self {
        let openai_client = openai::Client::from_env();
        let llm = openai_client
            .agent("gpt-4o-mini")
            .preamble(
                "You are an expert crypto analyst and prediction market trader.\n\n\
                 Your job: Given a Polymarket crypto question, current market price, price history, \
                 AND live crypto prices/news, estimate the TRUE probability that YES wins.\n\n\
                 You MUST consider:\n\
                 1. The LIVE crypto prices provided — these are real-time, use them\n\
                 2. The price trend from the market's history\n\
                 3. Time until expiry — shorter timeframes = less can change\n\
                 4. Whether the market is mispriced vs the live data\n\
                 5. Crypto volatility patterns — BTC can move 5-10% in a day\n\n\
                 IMPORTANT: Only output high confidence (>0.7) when live crypto data \
                 strongly supports your thesis. Be honest about uncertainty.\n\n\
                 Respond with ONLY a JSON object:\n\
                 {\"probability\": 0.XX, \"confidence\": 0.XX, \"reasoning\": \"one sentence why\"}\n\n\
                 - probability: true probability of YES outcome (0.01-0.99)\n\
                 - confidence: how sure you are (0.0 = guessing, 1.0 = certain)\n\
                 - reasoning: brief thesis based on the data",
            )
            .temperature(0.3)
            .build();

        Self {
            http: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            llm,
        }
    }

    /// Fetch live crypto prices from CoinGecko (free, no API key).
    async fn fetch_crypto_context(&self) -> Result<CryptoContext> {
        let url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true";

        let resp: serde_json::Value = self
            .http
            .get(url)
            .send()
            .await?
            .json()
            .await
            .context("failed to parse coingecko response")?;

        Ok(CryptoContext {
            btc_price: resp["bitcoin"]["usd"].as_f64().unwrap_or(0.0),
            eth_price: resp["ethereum"]["usd"].as_f64().unwrap_or(0.0),
            sol_price: resp["solana"]["usd"].as_f64().unwrap_or(0.0),
            btc_24h_change: resp["bitcoin"]["usd_24h_change"].as_f64().unwrap_or(0.0),
            eth_24h_change: resp["ethereum"]["usd_24h_change"].as_f64().unwrap_or(0.0),
            sol_24h_change: resp["solana"]["usd_24h_change"].as_f64().unwrap_or(0.0),
        })
    }

    /// Fetch active, open markets from Gamma API.
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

            if count < page_size || all.len() >= 500 {
                break;
            }
            offset += count;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(all)
    }

    /// Fetch recent price history for a token.
    pub async fn fetch_price_history(&self, token_id: &str) -> Result<Vec<PriceTick>> {
        let url = format!("{CLOB_API}/prices-history?market={token_id}&interval=max");

        #[derive(serde::Deserialize)]
        struct Resp {
            history: Vec<PriceTick>,
        }

        let resp: Resp = self
            .http
            .get(&url)
            .send()
            .await?
            .json()
            .await
            .context("failed to parse price history")?;

        Ok(resp.history)
    }

    /// Returns true if the market expires within MAX_DAYS_TO_EXPIRY.
    fn expires_within_window(end_date: Option<&str>) -> bool {
        let Some(date_str) = end_date else {
            return false;
        };
        let Ok(end) = date_str.parse::<DateTime<Utc>>() else {
            return false;
        };
        let now = Utc::now();
        let deadline = now + ChronoDuration::days(MAX_DAYS_TO_EXPIRY);
        end > now && end <= deadline
    }

    /// Ask the LLM to estimate probability given market context + live crypto data.
    async fn llm_estimate(
        &self,
        question: &str,
        current_price: f64,
        history: &[PriceTick],
        end_date: Option<&str>,
        crypto: &CryptoContext,
    ) -> Result<(f64, f64, String)> {
        let history_summary = if history.len() >= 5 {
            let recent: Vec<&PriceTick> = history.iter().rev().take(10).collect();
            let prices: Vec<String> = recent.iter().rev().map(|t| format!("{:.2}", t.p)).collect();

            let first = history.first().unwrap().p;
            let min = history.iter().map(|t| t.p).fold(f64::MAX, f64::min);
            let max = history.iter().map(|t| t.p).fold(f64::MIN, f64::max);
            let avg = history.iter().map(|t| t.p).sum::<f64>() / history.len() as f64;

            format!(
                "Recent prices: [{}]\nAll-time: min={min:.2}, max={max:.2}, avg={avg:.2}, start={first:.2}\nTotal ticks: {}",
                prices.join(", "),
                history.len()
            )
        } else {
            "Insufficient price history".to_string()
        };

        let expiry = end_date.unwrap_or("Unknown");
        let now: DateTime<Utc> = Utc::now();

        let prompt = format!(
            "{crypto_summary}\n\n\
             Market: \"{question}\"\n\
             Current YES price: {current_price:.4} ({pct:.1}%)\n\
             Expiry: {expiry}\n\
             Current time: {now}\n\n\
             Price history:\n{history_summary}\n\n\
             Using the LIVE crypto prices above, analyze this market and estimate the TRUE probability of YES.",
            crypto_summary = crypto.summary(),
            pct = current_price * 100.0,
        );

        let response = self.llm.chat(prompt, vec![]).await?;

        parse_llm_response(&response)
    }

    /// Scan all active crypto markets and return scored YES-only signals.
    pub async fn scan(&self) -> Result<Vec<Signal>> {
        // Fetch live crypto prices first
        let crypto = match self.fetch_crypto_context().await {
            Ok(c) => {
                tracing::info!(
                    btc = format_args!("${:.0}", c.btc_price),
                    eth = format_args!("${:.0}", c.eth_price),
                    sol = format_args!("${:.1}", c.sol_price),
                    "Live crypto prices"
                );
                c
            }
            Err(e) => {
                tracing::warn!(err = %e, "Failed to fetch crypto prices, using defaults");
                CryptoContext {
                    btc_price: 0.0,
                    eth_price: 0.0,
                    sol_price: 0.0,
                    btc_24h_change: 0.0,
                    eth_24h_change: 0.0,
                    sol_24h_change: 0.0,
                }
            }
        };

        let markets = self.fetch_active_markets().await?;
        tracing::info!(total = markets.len(), "Fetched active markets");

        // Filter: crypto, not noise, enough volume, has token, expires within 7 days
        let candidates: Vec<&GammaMarket> = markets
            .iter()
            .filter(|m| m.is_crypto_related())
            .filter(|m| !m.is_short_duration_noise())
            .filter(|m| m.volume_num >= MIN_VOLUME)
            .filter(|m| m.yes_token_id().is_some())
            .filter(|m| Self::expires_within_window(m.end_date.as_deref()))
            .collect();

        tracing::info!(
            crypto_candidates = candidates.len(),
            "Filtered crypto markets (expiring within {MAX_DAYS_TO_EXPIRY} days)"
        );

        let mut signals = Vec::new();

        for gm in &candidates {
            let token_id = gm.yes_token_id().unwrap();

            let current_price = match &gm.outcome_prices {
                Some(prices_str) => {
                    let prices: Vec<String> = serde_json::from_str(prices_str).unwrap_or_default();
                    prices
                        .first()
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0)
                }
                None => continue,
            };

            if current_price <= 0.03 || current_price >= 0.97 {
                continue;
            }

            // Fetch price history
            tokio::time::sleep(Duration::from_millis(150)).await;
            let history = match self.fetch_price_history(&token_id).await {
                Ok(h) if h.len() >= 10 => h,
                Ok(_) => continue,
                Err(e) => {
                    tracing::warn!(market = %gm.market_id, err = %e, "Failed to fetch history");
                    continue;
                }
            };

            // Ask LLM with live crypto context
            tokio::time::sleep(Duration::from_millis(200)).await;
            let (prob, confidence, reasoning) = match self
                .llm_estimate(
                    &gm.question,
                    current_price,
                    &history,
                    gm.end_date.as_deref(),
                    &crypto,
                )
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(market = %gm.market_id, err = %e, "LLM estimate failed");
                    continue;
                }
            };

            let edge = prob - current_price;

            // YES only — skip if LLM thinks NO is more likely
            if edge <= 0.0 {
                tracing::debug!(
                    market = %gm.question,
                    edge = format_args!("{:+.1}%", edge * 100.0),
                    "Skipping — no YES edge"
                );
                continue;
            }

            let kelly_size = fractional_kelly(prob, current_price, KELLY_FRACTION);
            let effective_edge = edge * confidence;

            tracing::info!(
                market = %gm.question,
                price = format_args!("{:.1}%", current_price * 100.0),
                est = format_args!("{:.1}%", prob * 100.0),
                edge = format_args!("+{:.1}%", edge * 100.0),
                conf = format_args!("{:.0}%", confidence * 100.0),
                kelly = format_args!("{:.1}%", kelly_size * 100.0),
                "Analyzed"
            );

            if effective_edge >= MIN_EDGE_PCT && kelly_size > 0.01 && confidence >= 0.5 {
                signals.push(Signal {
                    market_id: gm.market_id.clone(),
                    question: gm.question.clone(),
                    current_price,
                    estimated_prob: prob,
                    confidence,
                    edge,
                    kelly_size,
                    reasoning,
                    end_date: gm.end_date.clone(),
                    volume: gm.volume_num,
                });
            }
        }

        // Sort by score (best first)
        signals.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap());

        tracing::info!(
            signals = signals.len(),
            "Scan complete, high-conviction YES signals"
        );

        Ok(signals)
    }
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
