use serde::{Deserialize, Deserializer, Serialize};

fn string_to_f64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrFloat {
        String(String),
        Float(f64),
    }
    match StringOrFloat::deserialize(deserializer)? {
        StringOrFloat::String(s) => s.parse().map_err(serde::de::Error::custom),
        StringOrFloat::Float(f) => Ok(f),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Market {
    pub id: String,
    pub question: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub outcomes: Option<String>,
    #[serde(default)]
    pub outcome_prices: Option<String>,
    #[serde(default, deserialize_with = "string_to_f64")]
    pub volume: f64,
    #[serde(default, alias = "liquidityNum")]
    pub liquidity_num: f64,
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub closed: bool,
    #[serde(default)]
    pub end_date_iso: Option<String>,
}

impl Market {
    /// Parse outcome prices from the JSON string array, e.g. "[\"0.62\", \"0.38\"]"
    pub fn outcome_prices_parsed(&self) -> (f64, f64) {
        let Some(ref prices_str) = self.outcome_prices else {
            return (0.0, 0.0);
        };
        let prices: Vec<String> = serde_json::from_str(prices_str).unwrap_or_default();
        let yes = prices.first().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let no = prices.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
        (yes, no)
    }

    pub fn yes_price(&self) -> f64 {
        self.outcome_prices_parsed().0
    }

    pub fn no_price(&self) -> f64 {
        self.outcome_prices_parsed().1
    }

    pub fn implied_probability(&self) -> f64 {
        self.yes_price()
    }

    pub fn spread(&self) -> f64 {
        let (yes, no) = self.outcome_prices_parsed();
        (yes + no - 1.0).abs()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub market_id: String,
    pub bids: Vec<OrderLevel>,
    pub asks: Vec<OrderLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderLevel {
    pub price: f64,
    pub size: f64,
}
