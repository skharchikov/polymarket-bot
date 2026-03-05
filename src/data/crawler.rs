use anyhow::{Context, Result};
use reqwest::Client;
use std::time::Duration;
use tokio::time::sleep;

use super::models::{GammaMarket, HistoricalMarket, PriceHistoryResponse, PriceTick};

const GAMMA_API: &str = "https://gamma-api.polymarket.com";
const CLOB_API: &str = "https://clob.polymarket.com";

pub struct Crawler {
    client: Client,
    rate_limit_ms: u64,
}

impl Crawler {
    pub fn new(rate_limit_ms: u64) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            rate_limit_ms,
        }
    }

    /// Fetch resolved markets from Gamma API.
    pub async fn fetch_resolved_markets(&self, limit: usize) -> Result<Vec<GammaMarket>> {
        let url = format!("{GAMMA_API}/markets?closed=true&limit={limit}");
        let markets: Vec<GammaMarket> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await
            .context("failed to parse gamma markets")?;
        Ok(markets)
    }

    /// Fetch price history for a single CLOB token.
    pub async fn fetch_price_history(&self, token_id: &str) -> Result<Vec<PriceTick>> {
        let url = format!("{CLOB_API}/prices-history?market={token_id}&interval=max");
        let resp: PriceHistoryResponse = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await
            .context("failed to parse price history")?;
        Ok(resp.history)
    }

    /// Crawl resolved markets and build historical dataset.
    pub async fn build_dataset(&self, market_limit: usize) -> Result<Vec<HistoricalMarket>> {
        let gamma_markets = self.fetch_resolved_markets(market_limit).await?;
        tracing::info!(count = gamma_markets.len(), "Fetched resolved markets");

        let mut dataset = Vec::new();

        for gm in &gamma_markets {
            let resolved_yes = match gm.outcome.as_deref() {
                Some("Yes") => true,
                Some("No") => false,
                _ => continue, // skip unresolved
            };

            let token_ids = match &gm.clob_token_ids {
                Some(ids) if !ids.is_empty() => ids.clone(),
                _ => continue,
            };

            // Parse comma-separated token IDs, take the first (YES token)
            let token_id = token_ids
                .split(',')
                .next()
                .unwrap_or_default()
                .trim()
                .to_string();

            if token_id.is_empty() {
                continue;
            }

            let end_date = match &gm.end_date {
                Some(d) => match d.parse() {
                    Ok(dt) => dt,
                    Err(_) => continue,
                },
                None => continue,
            };

            sleep(Duration::from_millis(self.rate_limit_ms)).await;

            match self.fetch_price_history(&token_id).await {
                Ok(history) if !history.is_empty() => {
                    tracing::info!(
                        market = %gm.market_id,
                        ticks = history.len(),
                        "Downloaded price history"
                    );
                    dataset.push(HistoricalMarket {
                        market_id: gm.market_id.clone(),
                        question: gm.question.clone(),
                        token_id,
                        end_date,
                        resolved_yes,
                        price_history: history,
                    });
                }
                Ok(_) => {
                    tracing::debug!(market = %gm.market_id, "Empty price history, skipping");
                }
                Err(e) => {
                    tracing::warn!(market = %gm.market_id, err = %e, "Failed to fetch history");
                }
            }
        }

        tracing::info!(total = dataset.len(), "Historical dataset built");
        Ok(dataset)
    }
}
