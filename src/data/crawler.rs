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

    /// Fetch resolved markets from Gamma API, newest first.
    pub async fn fetch_resolved_markets(&self, limit: usize) -> Result<Vec<GammaMarket>> {
        let url = format!(
            "{GAMMA_API}/markets?closed=true&limit={limit}&order=updatedAt&ascending=false"
        );
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
    /// If `crypto_only` is true, only include crypto-related markets.
    pub async fn build_dataset(
        &self,
        market_limit: usize,
        crypto_only: bool,
    ) -> Result<Vec<HistoricalMarket>> {
        let gamma_markets = self.fetch_resolved_markets(market_limit).await?;
        tracing::info!(count = gamma_markets.len(), "Fetched resolved markets");

        let mut dataset = Vec::new();

        for gm in &gamma_markets {
            if crypto_only && !gm.is_crypto_related() {
                continue;
            }

            let resolved_yes = match gm.resolved_yes() {
                Some(v) => v,
                None => continue,
            };

            let token_id = match gm.yes_token_id() {
                Some(id) => id,
                None => continue,
            };

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
                        question = %gm.question,
                        ticks = history.len(),
                        resolved_yes,
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
