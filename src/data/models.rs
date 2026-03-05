use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single price tick from CLOB history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceTick {
    /// Unix timestamp
    pub t: i64,
    /// Price (0.0 - 1.0)
    pub p: f64,
}

/// Response from the CLOB prices-history endpoint.
#[derive(Debug, Deserialize)]
pub struct PriceHistoryResponse {
    pub history: Vec<PriceTick>,
}

/// A fully resolved market with its history, ready for backtesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalMarket {
    pub market_id: String,
    pub question: String,
    pub token_id: String,
    pub end_date: DateTime<Utc>,
    /// true = YES won, false = NO won
    pub resolved_yes: bool,
    pub price_history: Vec<PriceTick>,
}

/// Compact market info from Gamma API for crawling.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GammaMarket {
    #[serde(rename = "id")]
    pub market_id: String,
    pub question: String,
    #[serde(default)]
    pub clob_token_ids: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    /// JSON array string, e.g. "[\"1\", \"0\"]" — first element is YES price
    #[serde(default)]
    pub outcome_prices: Option<String>,
    /// JSON array string, e.g. "[\"Yes\", \"No\"]"
    #[serde(default)]
    pub outcomes: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
}

impl GammaMarket {
    /// Determine resolution from outcomePrices: ["1","0"] = YES won, ["0","1"] = NO won.
    pub fn resolved_yes(&self) -> Option<bool> {
        let prices_str = self.outcome_prices.as_ref()?;
        let prices: Vec<String> = serde_json::from_str(prices_str).ok()?;
        let yes_price: f64 = prices.first()?.parse().ok()?;
        let no_price: f64 = prices.get(1)?.parse().ok()?;

        if yes_price == 1.0 && no_price == 0.0 {
            Some(true)
        } else if yes_price == 0.0 && no_price == 1.0 {
            Some(false)
        } else {
            None // not fully resolved
        }
    }

    /// Parse clobTokenIds JSON array, return first token (YES token).
    pub fn yes_token_id(&self) -> Option<String> {
        let ids_str = self.clob_token_ids.as_ref()?;
        let ids: Vec<String> = serde_json::from_str(ids_str).ok()?;
        ids.into_iter().next().filter(|s| !s.is_empty())
    }

    pub fn is_crypto_related(&self) -> bool {
        let q = self.question.to_lowercase();
        let cat = self.category.as_deref().unwrap_or_default().to_lowercase();

        const KEYWORDS: &[&str] = &[
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "solana",
            "sol",
            "crypto",
            "token",
            "defi",
            "nft",
            "blockchain",
            "altcoin",
            "dogecoin",
            "doge",
            "xrp",
            "ripple",
            "cardano",
            "ada",
            "polkadot",
            "avalanche",
            "avax",
            "chainlink",
            "link",
            "bnb",
            "binance",
            "coinbase",
            "stablecoin",
            "usdc",
            "usdt",
            "tether",
            "memecoin",
        ];

        KEYWORDS.iter().any(|kw| q.contains(kw)) || cat.contains("crypto")
    }
}
