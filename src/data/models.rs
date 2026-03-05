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
pub struct GammaMarket {
    #[serde(rename = "id")]
    pub market_id: String,
    pub question: String,
    #[serde(rename = "clobTokenIds")]
    pub clob_token_ids: Option<String>,
    #[serde(rename = "endDate")]
    pub end_date: Option<String>,
    /// "Yes" or "No" or empty if unresolved
    #[serde(rename = "outcome")]
    pub outcome: Option<String>,
    pub active: Option<bool>,
    pub closed: Option<bool>,
}
