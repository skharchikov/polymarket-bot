use anyhow::{Context, Result};
use reqwest::Client;
use std::time::Duration;
use tokio::time::sleep;

use super::models::{GammaMarket, HistoricalMarket, PriceHistoryResponse, PriceTick};

const GAMMA_API: &str = "https://gamma-api.polymarket.com";
const CLOB_API: &str = "https://clob.polymarket.com";
const PAGE_SIZE: usize = 100;

pub struct CrawlerConfig {
    pub rate_limit_ms: u64,
    pub market_limit: usize,
    pub crypto_only: bool,
    pub min_volume: f64,
    pub min_ticks: usize,
    pub min_duration_hours: f64,
}

impl Default for CrawlerConfig {
    fn default() -> Self {
        Self {
            rate_limit_ms: 100,
            market_limit: 2000,
            crypto_only: true,
            min_volume: 1000.0,
            min_ticks: 20,
            min_duration_hours: 4.0,
        }
    }
}

pub struct Crawler {
    client: Client,
    config: CrawlerConfig,
}

impl Crawler {
    pub fn new(config: CrawlerConfig) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            config,
        }
    }

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

    async fn fetch_resolved_markets(&self) -> Result<Vec<GammaMarket>> {
        let total = self.config.market_limit;
        let mut all = Vec::new();
        let mut offset = 0;

        while all.len() < total {
            let page_size = PAGE_SIZE.min(total - all.len());
            let page = self.fetch_page(offset, page_size).await?;
            let count = page.len();
            all.extend(page);
            tracing::info!(fetched = all.len(), target = total, "Fetching markets...");

            if count < page_size {
                break;
            }
            offset += count;
            sleep(Duration::from_millis(100)).await;
        }

        Ok(all)
    }

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

    pub async fn build_dataset(&self) -> Result<Vec<HistoricalMarket>> {
        let gamma_markets = self.fetch_resolved_markets().await?;
        tracing::info!(count = gamma_markets.len(), "Fetched resolved markets");

        let candidates: Vec<_> = gamma_markets
            .iter()
            .filter(|gm| !gm.is_short_duration_noise())
            .filter(|gm| !self.config.crypto_only || gm.is_crypto_related())
            .filter(|gm| gm.volume_num >= self.config.min_volume)
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
            "Eligible markets (filtered noise + min volume ${:.0})",
            self.config.min_volume
        );

        let mut dataset = Vec::new();

        for gm in &candidates {
            let resolved_yes = gm.resolved_yes().unwrap();
            let token_id = gm.yes_token_id().unwrap();
            let end_date = gm.end_date.as_ref().unwrap().parse().unwrap();

            sleep(Duration::from_millis(self.config.rate_limit_ms)).await;

            match self.fetch_price_history(&token_id).await {
                Ok(history) if history.len() >= self.config.min_ticks => {
                    let market = HistoricalMarket {
                        market_id: gm.market_id.clone(),
                        question: gm.question.clone(),
                        token_id,
                        end_date,
                        resolved_yes,
                        price_history: history,
                    };

                    if market.duration_hours() >= self.config.min_duration_hours {
                        dataset.push(market);
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(market = %gm.market_id, err = %e, "Failed to fetch history");
                }
            }

            if dataset.len() % 25 == 0 && !dataset.is_empty() {
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
