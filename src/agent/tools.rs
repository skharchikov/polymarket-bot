use rig::tool::Tool;
use serde::{Deserialize, Serialize};

use crate::markets::fetcher::MarketFetcher;
use crate::markets::models::Market;

#[derive(Deserialize)]
pub struct GetMarketsArgs {
    pub limit: Option<usize>,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ToolError(String);

#[derive(Serialize)]
pub struct MarketList {
    pub markets: Vec<MarketSummary>,
}

#[derive(Serialize)]
pub struct MarketSummary {
    pub id: String,
    pub question: String,
    pub yes_price: f64,
    pub liquidity: f64,
}

pub struct GetMarketsTool {
    fetcher: MarketFetcher,
}

impl GetMarketsTool {
    pub fn new(fetcher: MarketFetcher) -> Self {
        Self { fetcher }
    }
}

impl Tool for GetMarketsTool {
    const NAME: &'static str = "get_markets";

    type Args = GetMarketsArgs;
    type Output = MarketList;
    type Error = ToolError;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let limit = args.limit.unwrap_or(20);
        let markets = self
            .fetcher
            .fetch_markets(limit)
            .await
            .map_err(|e| ToolError(e.to_string()))?;

        Ok(MarketList {
            markets: markets
                .into_iter()
                .map(|m| MarketSummary {
                    id: m.id,
                    question: m.question,
                    yes_price: m.yes_price,
                    liquidity: m.liquidity,
                })
                .collect(),
        })
    }

    fn definition(&self, _name: String) -> rig::tool::ToolDefinition {
        rig::tool::ToolDefinition {
            name: "get_markets".to_string(),
            description: "Fetch active prediction markets from Polymarket".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of markets to fetch (default 20)"
                    }
                }
            }),
        }
    }
}
