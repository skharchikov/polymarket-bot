use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub id: String,
    pub question: String,
    pub description: String,
    pub yes_price: f64,
    pub no_price: f64,
    pub liquidity: f64,
    pub volume: f64,
    pub end_date: DateTime<Utc>,
    pub active: bool,
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

impl Market {
    pub fn implied_probability(&self) -> f64 {
        self.yes_price
    }

    pub fn spread(&self) -> f64 {
        (self.yes_price + self.no_price - 1.0).abs()
    }
}
