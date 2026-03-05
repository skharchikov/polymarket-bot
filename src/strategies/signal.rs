use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Side {
    Yes,
    No,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub market_id: String,
    pub side: Side,
    pub edge: f64,
    pub confidence: f64,
    pub size_pct: f64,
    pub source: SignalSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalSource {
    Value { model_prob: f64, market_prob: f64 },
    Arbitrage { paired_market: String, spread: f64 },
    Momentum { trend: f64, lookback: usize },
}
