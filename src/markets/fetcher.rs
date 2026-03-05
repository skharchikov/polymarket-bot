use anyhow::Result;
use reqwest::Client;

use super::models::Market;

const POLYMARKET_API: &str = "https://gamma-api.polymarket.com";

pub struct MarketFetcher {
    client: Client,
}

impl MarketFetcher {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub async fn fetch_markets(&self, limit: usize) -> Result<Vec<Market>> {
        let url = format!("{POLYMARKET_API}/markets?limit={limit}&active=true");
        let markets: Vec<Market> = self.client.get(&url).send().await?.json().await?;
        Ok(markets)
    }

    pub async fn fetch_market(&self, id: &str) -> Result<Market> {
        let url = format!("{POLYMARKET_API}/markets/{id}");
        let market: Market = self.client.get(&url).send().await?.json().await?;
        Ok(market)
    }
}
