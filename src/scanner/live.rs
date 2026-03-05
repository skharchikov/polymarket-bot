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

/// Minimum technical edge before we bother calling the LLM.
const MIN_TECHNICAL_EDGE: f64 = 0.08;
/// Minimum final edge (after LLM confirmation) to emit a signal.
const MIN_FINAL_EDGE: f64 = 0.10;
const MIN_VOLUME: f64 = 5000.0;
const KELLY_FRACTION: f64 = 0.25;
const MAX_DAYS_TO_EXPIRY: i64 = 7;
/// Max markets to send to LLM per scan (cost control).
const MAX_LLM_CALLS: usize = 3;

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

/// Pre-LLM candidate with technical analysis score.
#[derive(Debug)]
struct TechnicalCandidate {
    market: GammaMarket,
    token_id: String,
    current_price: f64,
    history: Vec<PriceTick>,
    tech_prob: f64,
    tech_edge: f64,
}

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
                 AND live crypto prices, estimate the TRUE probability that YES wins.\n\n\
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

    async fn fetch_crypto_context(&self) -> Result<CryptoContext> {
        let url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true";
        let resp: serde_json::Value = self.http.get(url).send().await?.json().await?;
        Ok(CryptoContext {
            btc_price: resp["bitcoin"]["usd"].as_f64().unwrap_or(0.0),
            eth_price: resp["ethereum"]["usd"].as_f64().unwrap_or(0.0),
            sol_price: resp["solana"]["usd"].as_f64().unwrap_or(0.0),
            btc_24h_change: resp["bitcoin"]["usd_24h_change"].as_f64().unwrap_or(0.0),
            eth_24h_change: resp["ethereum"]["usd_24h_change"].as_f64().unwrap_or(0.0),
            sol_24h_change: resp["solana"]["usd_24h_change"].as_f64().unwrap_or(0.0),
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

            if count < page_size || all.len() >= 500 {
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

    /// Pure technical analysis — no LLM call. Returns estimated prob of YES.
    fn technical_estimate(history: &[PriceTick], current_price: f64) -> f64 {
        let n = history.len();
        if n < 5 {
            return current_price;
        }

        let prices: Vec<f64> = history.iter().map(|t| t.p).collect();

        // 1. SMA momentum: short vs long average
        let short_w = (n / 4).max(3).min(n);
        let sma_short: f64 = prices[n - short_w..].iter().sum::<f64>() / short_w as f64;
        let sma_long: f64 = prices.iter().sum::<f64>() / n as f64;
        let momentum = sma_short - sma_long;

        // 2. Trend: linear slope
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean = sma_long;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &p) in prices.iter().enumerate() {
            let x = i as f64 - x_mean;
            num += x * (p - y_mean);
            den += x * x;
        }
        let slope = if den > 0.0 { num / den } else { 0.0 };

        // 3. Recent volatility
        let recent = &prices[n.saturating_sub(10)..];
        let recent_mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let vol: f64 = (recent
            .iter()
            .map(|p| (p - recent_mean).powi(2))
            .sum::<f64>()
            / recent.len() as f64)
            .sqrt();

        // Combine signals: trend + momentum, dampened by volatility
        let vol_factor = if vol > 0.15 {
            0.3 // high vol = less trust in signals
        } else if vol < 0.02 {
            1.0 // low vol = strong signal
        } else {
            0.7
        };

        let signal = (momentum * 2.0 + slope * n as f64 * 0.3) * vol_factor;

        (current_price + signal).clamp(0.02, 0.98)
    }

    /// LLM call — only for top technical candidates.
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
            "{crypto}\n\n\
             Market: \"{question}\"\n\
             Current YES price: {current_price:.4} ({pct:.1}%)\n\
             Expiry: {expiry}\n\
             Current time: {now}\n\n\
             Price history:\n{history_summary}\n\n\
             Using the LIVE crypto prices above, estimate TRUE probability of YES.",
            crypto = crypto.summary(),
            pct = current_price * 100.0,
        );

        let response = self.llm.chat(prompt, vec![]).await?;
        parse_llm_response(&response)
    }

    /// Main scan: technical filter → top N → LLM confirmation → signals.
    pub async fn scan(&self, skip_market_ids: &[String]) -> Result<Vec<Signal>> {
        // 1. Fetch live crypto prices (free)
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
                tracing::warn!(err = %e, "Failed to fetch crypto prices");
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

        // 2. Fetch and filter markets (free)
        let markets = self.fetch_active_markets().await?;
        tracing::info!(total = markets.len(), "Fetched active markets");

        let candidates: Vec<&GammaMarket> = markets
            .iter()
            .filter(|m| m.is_crypto_related())
            .filter(|m| !m.is_short_duration_noise())
            .filter(|m| m.volume_num >= MIN_VOLUME)
            .filter(|m| m.yes_token_id().is_some())
            .filter(|m| Self::expires_within_window(m.end_date.as_deref()))
            .filter(|m| !skip_market_ids.contains(&m.market_id))
            .collect();

        tracing::info!(candidates = candidates.len(), "Crypto markets (≤7 days)");

        // 3. Technical analysis on each candidate (free — just math)
        let mut tech_candidates: Vec<TechnicalCandidate> = Vec::new();

        for gm in &candidates {
            let token_id = gm.yes_token_id().unwrap();
            let current_price = match &gm.outcome_prices {
                Some(s) => {
                    let p: Vec<String> = serde_json::from_str(s).unwrap_or_default();
                    p.first().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0)
                }
                None => continue,
            };

            if current_price <= 0.05 || current_price >= 0.95 {
                continue;
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
            let history = match self.fetch_price_history(&token_id).await {
                Ok(h) if h.len() >= 10 => h,
                Ok(_) => continue,
                Err(e) => {
                    tracing::warn!(market = %gm.market_id, err = %e, "History fetch failed");
                    continue;
                }
            };

            let tech_prob = Self::technical_estimate(&history, current_price);
            let tech_edge = tech_prob - current_price;

            tracing::info!(
                market = %gm.question,
                price = format_args!("{:.1}%", current_price * 100.0),
                tech = format_args!("{:.1}%", tech_prob * 100.0),
                edge = format_args!("{:+.1}%", tech_edge * 100.0),
                "Technical analysis"
            );

            // Only keep YES-edge candidates above threshold
            if tech_edge >= MIN_TECHNICAL_EDGE {
                tech_candidates.push(TechnicalCandidate {
                    market: (*gm).clone(),
                    token_id,
                    current_price,
                    history,
                    tech_prob,
                    tech_edge,
                });
            }
        }

        // Sort by technical edge, take top N for LLM
        tech_candidates.sort_by(|a, b| b.tech_edge.partial_cmp(&a.tech_edge).unwrap());
        let top = tech_candidates.into_iter().take(MAX_LLM_CALLS);

        tracing::info!("Sending top candidates to LLM for confirmation...");

        // 4. LLM confirmation only on best technical candidates (paid)
        let mut signals = Vec::new();

        for tc in top {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let (prob, confidence, reasoning) = match self
                .llm_estimate(
                    &tc.market.question,
                    tc.current_price,
                    &tc.history,
                    tc.market.end_date.as_deref(),
                    &crypto,
                )
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(market = %tc.market.market_id, err = %e, "LLM failed");
                    continue;
                }
            };

            let edge = prob - tc.current_price;

            if edge <= 0.0 {
                tracing::info!(
                    market = %tc.market.question,
                    llm_prob = format_args!("{:.1}%", prob * 100.0),
                    "LLM disagrees — no YES edge"
                );
                continue;
            }

            let kelly_size = fractional_kelly(prob, tc.current_price, KELLY_FRACTION);
            let effective_edge = edge * confidence;

            tracing::info!(
                market = %tc.market.question,
                price = format_args!("{:.1}%", tc.current_price * 100.0),
                llm = format_args!("{:.1}%", prob * 100.0),
                edge = format_args!("+{:.1}%", edge * 100.0),
                conf = format_args!("{:.0}%", confidence * 100.0),
                kelly = format_args!("{:.1}%", kelly_size * 100.0),
                "LLM confirmed"
            );

            if effective_edge >= MIN_FINAL_EDGE && kelly_size > 0.01 && confidence >= 0.5 {
                signals.push(Signal {
                    market_id: tc.market.market_id,
                    question: tc.market.question,
                    current_price: tc.current_price,
                    estimated_prob: prob,
                    confidence,
                    edge,
                    kelly_size,
                    reasoning,
                    end_date: tc.market.end_date,
                    volume: tc.market.volume_num,
                });
            }
        }

        signals.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap());

        tracing::info!(signals = signals.len(), "Final high-conviction signals");
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
