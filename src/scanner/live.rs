use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use reqwest::Client;
use rig::completion::Chat;
use rig::providers::openai;
use std::time::Duration;

use crate::data::models::{GammaMarket, PriceTick};
use crate::pricing::kelly::fractional_kelly;
use crate::storage::portfolio::{BetContext, BetSide};

use super::news::{NewsAggregator, NewsItem, NewsMatch};

const GAMMA_API: &str = "https://gamma-api.polymarket.com";
const CLOB_API: &str = "https://clob.polymarket.com";

const MIN_VOLUME: f64 = 5000.0;
const MIN_BOOK_DEPTH: f64 = 200.0;
const KELLY_FRACTION: f64 = 0.25;
const MAX_DAYS_TO_EXPIRY: i64 = 14;
/// Max markets to send to LLM per scan cycle.
const MAX_LLM_CANDIDATES: usize = 3;
/// Minimum edge (prob delta * confidence) to emit a signal.
const MIN_EFFECTIVE_EDGE: f64 = 0.08;

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
        format!(
            "{emoji} *{side_label} Signal*\n\n\
             \u{1f4cb} *{question}*\n\n\
             \u{1f4b0} Current price: `{price:.1}\u{00a2}`\n\
             \u{1f3af} Our estimate: `{est:.1}%`\n\
             \u{1f4ca} Edge: `+{edge:.1}%`\n\
             \u{1f512} Confidence: `{conf:.0}%`\n\
             \u{1f4d0} Kelly size: `{kelly:.1}%` of bankroll\n\
             \u{1f4a7} Volume: `${vol:.0}`\n\
             \u{23f0} Expires: {end}\n\n\
             \u{1f4a1} _{reasoning}_",
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

pub struct LiveScanner {
    http: Client,
    openai_client: openai::Client,
    news: NewsAggregator,
}

impl LiveScanner {
    pub fn new() -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client");
        Self {
            news: NewsAggregator::new(http.clone()),
            http,
            openai_client: openai::Client::from_env(),
        }
    }

    /// Build an LLM agent for news-impact assessment.
    fn build_agent(&self, temperature: f64) -> rig::agent::Agent<openai::CompletionModel> {
        self.openai_client
            .agent("gpt-4o-mini")
            .preamble(
                "You are an expert prediction market analyst.\n\n\
                 Your job: Given a Polymarket prediction market and RECENT NEWS, assess whether \
                 the news changes the probability of YES winning.\n\n\
                 KEY PRINCIPLE: You are looking for NEWS that the market has NOT YET priced in.\n\
                 If the news is old or already reflected in the price, say so (low confidence).\n\
                 If the news is fresh and clearly impacts the outcome, say so (high confidence).\n\n\
                 You can profit from BOTH sides:\n\
                 - If news makes YES more likely than the current price suggests -> buy YES\n\
                 - If news makes YES less likely than the current price suggests -> buy NO\n\n\
                 Be honest and calibrated. Only give high confidence (>0.7) when:\n\
                 1. The news is directly relevant to this specific market\n\
                 2. The news clearly shifts the probability\n\
                 3. The current price doesn't already reflect this news\n\n\
                 Respond with ONLY a JSON object:\n\
                 {\"probability\": 0.XX, \"confidence\": 0.XX, \"reasoning\": \"one sentence\"}\n\n\
                 - probability: true probability of YES (0.01-0.99)\n\
                 - confidence: 0.0 (no useful info) to 1.0 (news clearly changes outcome)\n\
                 - reasoning: explain what news matters and why price should move",
            )
            .temperature(temperature)
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

    fn get_yes_price(gm: &GammaMarket) -> Option<f64> {
        let s = gm.outcome_prices.as_ref()?;
        let p: Vec<String> = serde_json::from_str(s).ok()?;
        p.first().and_then(|s| s.parse::<f64>().ok())
    }

    /// Ask LLM: given this market + news, what's the true probability?
    /// Uses 2-call consensus (speed/cost balance for news-driven approach).
    async fn assess_news_impact(
        &self,
        market: &GammaMarket,
        current_price: f64,
        news: &[NewsItem],
        history_summary: &str,
        past_bets: &str,
    ) -> Result<(f64, f64, String)> {
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

        // Single LLM call (respects 3 RPM rate limit on Tier-1 OpenAI)
        let agent = self.build_agent(0.2);
        let response = agent
            .chat(prompt.as_str(), vec![])
            .await
            .map_err(|e| anyhow::anyhow!("LLM call failed: {e}"))?;
        let (prob, conf, reasoning) = parse_llm_response(&response)?;

        tracing::info!(
            market = %market.question,
            prob = format_args!("{:.1}%", prob * 100.0),
            conf = format_args!("{:.0}%", conf * 100.0),
            "News impact assessment"
        );

        Ok((prob, conf, reasoning))
    }

    /// Main scan: fetch news → match to markets → LLM assesses impact → signals.
    pub async fn scan(
        &self,
        skip_market_ids: &[String],
        past_bets_summary: &str,
    ) -> Result<Vec<Signal>> {
        // Step 1: Fetch all active markets (ALL categories, not just crypto)
        let markets = self.fetch_active_markets().await?;
        tracing::info!(total = markets.len(), "Fetched active markets");

        let eligible: Vec<GammaMarket> = markets
            .into_iter()
            .filter(|m| m.volume_num >= MIN_VOLUME)
            .filter(|m| m.yes_token_id().is_some())
            .filter(|m| Self::expires_within_window(m.end_date.as_deref()))
            .filter(|m| !skip_market_ids.contains(&m.market_id))
            .filter(|m| {
                let price = Self::get_yes_price(m).unwrap_or(0.0);
                price > 0.05 && price < 0.95
            })
            .collect();

        tracing::info!(
            eligible = eligible.len(),
            "Eligible markets (vol>${}, ≤{}d expiry)",
            MIN_VOLUME,
            MAX_DAYS_TO_EXPIRY,
        );

        // Step 2: Fetch breaking news from all sources
        let news = self.news.fetch_all().await;

        if news.is_empty() {
            tracing::warn!("No news fetched — check network/API access");
            return Ok(Vec::new());
        }

        // Step 3: Match news to markets by keyword relevance
        let matches = NewsAggregator::match_to_markets(&news, &eligible);

        tracing::info!(
            matched = matches.len(),
            "Markets matched with relevant news"
        );

        if matches.is_empty() {
            return Ok(Vec::new());
        }

        // Step 4: For top matches, check book depth and get price history
        let mut candidates: Vec<(NewsMatch, f64, String, f64)> = Vec::new();

        for nm in matches.iter().take(MAX_LLM_CANDIDATES * 2) {
            let token_id = nm.market.yes_token_id().unwrap();
            let current_price = Self::get_yes_price(&nm.market).unwrap_or(0.0);

            // Check liquidity
            tokio::time::sleep(Duration::from_millis(100)).await;
            let book_depth = self.fetch_book_depth(&token_id).await.unwrap_or(0.0);

            if book_depth < MIN_BOOK_DEPTH {
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

            if candidates.len() >= MAX_LLM_CANDIDATES {
                break;
            }
        }

        if candidates.is_empty() {
            tracing::info!("No news-matched markets passed liquidity filter");
            return Ok(Vec::new());
        }

        tracing::info!(
            count = candidates.len(),
            "Assessing news impact with LLM..."
        );

        // Step 5: LLM assesses news impact on each candidate
        let mut signals = Vec::new();

        for (i, (nm, current_price, history_summary, book_depth)) in candidates.iter().enumerate() {
            // Rate limit: wait 25s between LLM calls (3 RPM on Tier-1)
            if i > 0 {
                tracing::info!("Waiting 25s for rate limit...");
                tokio::time::sleep(Duration::from_secs(25)).await;
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
                tracing::info!(
                    market = %nm.market.question,
                    prob = format_args!("{:.1}%", prob * 100.0),
                    price = format_args!("{:.1}%", current_price * 100.0),
                    "No edge on either side"
                );
                continue;
            };

            let kelly_size = fractional_kelly(bet_prob, bet_price, KELLY_FRACTION);
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

            if effective_edge >= MIN_EFFECTIVE_EDGE && kelly_size > 0.01 && confidence >= 0.5 {
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
        }

        signals.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap());

        tracing::info!(signals = signals.len(), "Final news-driven signals");
        Ok(signals)
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
