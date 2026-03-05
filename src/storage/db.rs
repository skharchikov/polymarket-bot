use std::collections::HashMap;

use crate::strategies::signal::Signal;

pub struct TradeLog {
    trades: Vec<TradeRecord>,
}

pub struct TradeRecord {
    pub signal: Signal,
    pub amount_usd: f64,
    pub tx_hash: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl TradeLog {
    pub fn new() -> Self {
        Self { trades: Vec::new() }
    }

    pub fn record(&mut self, signal: Signal, amount_usd: f64, tx_hash: String) {
        self.trades.push(TradeRecord {
            signal,
            amount_usd,
            tx_hash,
            timestamp: chrono::Utc::now(),
        });
    }

    pub fn exposure_by_market(&self) -> HashMap<String, f64> {
        let mut exposure: HashMap<String, f64> = HashMap::new();
        for trade in &self.trades {
            *exposure.entry(trade.signal.market_id.clone()).or_default() += trade.amount_usd;
        }
        exposure
    }
}
