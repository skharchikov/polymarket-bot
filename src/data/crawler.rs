use anyhow::{Context, Result};
use reqwest::Client;
use std::time::Duration;
use tokio::time::sleep;

use super::models::{GammaMarket, HistoricalMarket, PriceHistoryResponse, PriceTick};

const GAMMA_API: &str = "https://gamma-api.polymarket.com";
const CLOB_API: &str = "https://clob.polymarket.com";
const PAGE_SIZE: usize = 100;

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

    /// Fetch a single page of resolved markets.
    async fn fetch_page(&self, offset: usize, limit: usize) -> Result<Vec<GammaMarket>> {
        let url = format!(
            "{GAMMA_API}/markets?closed=true&limit={limit}&offset={offset}&order=updatedAt&ascending=false"
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

    /// Paginate through resolved markets up to `total` count.
    pub async fn fetch_resolved_markets(&self, total: usize) -> Result<Vec<GammaMarket>> {
        let mut all = Vec::new();
        let mut offset = 0;

        while all.len() < total {
            let page_size = PAGE_SIZE.min(total - all.len());
            let page = self.fetch_page(offset, page_size).await?;
            let count = page.len();
            all.extend(page);
            tracing::info!(fetched = all.len(), target = total, "Fetching markets...");

            if count < page_size {
                break; // no more data
            }
            offset += count;
            sleep(Duration::from_millis(100)).await;
        }

        Ok(all)
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
    pub async fn build_dataset(
        &self,
        market_limit: usize,
        crypto_only: bool,
    ) -> Result<Vec<HistoricalMarket>> {
        let gamma_markets = self.fetch_resolved_markets(market_limit).await?;
        tracing::info!(count = gamma_markets.len(), "Fetched resolved markets");

        let candidates: Vec<_> = gamma_markets
            .iter()
            .filter(|gm| !crypto_only || gm.is_crypto_related())
            .filter(|gm| gm.resolved_yes().is_some())
            .filter(|gm| gm.yes_token_id().is_some())
            .filter(|gm| {
                gm.end_date
                    .as_ref()
                    .and_then(|d| d.parse::<chrono::DateTime<chrono::Utc>>().ok())
                    .is_some()
            })
            .collect();

        tracing::info!(
            candidates = candidates.len(),
            "Eligible markets, downloading price history..."
        );

        let mut dataset = Vec::new();

        for gm in &candidates {
            let resolved_yes = gm.resolved_yes().unwrap();
            let token_id = gm.yes_token_id().unwrap();
            let end_date = gm.end_date.as_ref().unwrap().parse().unwrap();

            sleep(Duration::from_millis(self.rate_limit_ms)).await;

            match self.fetch_price_history(&token_id).await {
                Ok(history) if !history.is_empty() => {
                    tracing::debug!(
                        market = %gm.market_id,
                        ticks = history.len(),
                        "[{}/{}] Downloaded",
                        dataset.len() + 1,
                        candidates.len()
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
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(market = %gm.market_id, err = %e, "Failed to fetch history");
                }
            }

            // Progress log every 50 markets
            if dataset.len() % 50 == 0 && !dataset.is_empty() {
                tracing::info!(
                    downloaded = dataset.len(),
                    total = candidates.len(),
                    "Progress"
                );
            }
        }

        tracing::info!(total = dataset.len(), "Historical dataset built");
        Ok(dataset)
    }
}
