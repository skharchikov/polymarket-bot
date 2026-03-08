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

impl HistoricalMarket {
    /// Duration of the market in hours based on first and last price tick.
    pub fn duration_hours(&self) -> f64 {
        if self.price_history.len() < 2 {
            return 0.0;
        }
        let first = self.price_history.first().unwrap().t;
        let last = self.price_history.last().unwrap().t;
        (last - first) as f64 / 3600.0
    }
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
    #[serde(default)]
    pub outcome_prices: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub outcomes: Option<String>,
    #[serde(default)]
    pub slug: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub volume_num: f64,
    #[serde(default)]
    #[allow(dead_code)]
    pub liquidity_num: f64,
}

impl GammaMarket {
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
            None
        }
    }

    pub fn yes_token_id(&self) -> Option<String> {
        let ids_str = self.clob_token_ids.as_ref()?;
        let ids: Vec<String> = serde_json::from_str(ids_str).ok()?;
        ids.into_iter().next().filter(|s| !s.is_empty())
    }

    pub fn polymarket_url(&self) -> String {
        match &self.slug {
            Some(slug) => format!("https://polymarket.com/event/{slug}"),
            None => format!("https://polymarket.com/event/{}", self.market_id),
        }
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
            "defi",
            "nft",
            "blockchain",
            "dogecoin",
            "doge",
            "xrp",
            "ripple",
            "cardano",
            "polkadot",
            "avalanche",
            "chainlink",
            "bnb",
            "binance",
            "coinbase",
            "stablecoin",
            "memecoin",
        ];

        KEYWORDS.iter().any(|kw| q.contains(kw)) || cat.contains("crypto")
    }

    /// Returns true for short-duration noise markets (5-min up/down, etc).
    pub fn is_short_duration_noise(&self) -> bool {
        let q = self.question.to_lowercase();
        q.contains("up or down")
    }
}
