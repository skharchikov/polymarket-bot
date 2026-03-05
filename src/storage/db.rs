use std::collections::HashMap;

use crate::strategies::mispricing::TradeSignal;

/// Simple in-memory trade log. Replace with SQLite/Postgres for persistence.
pub struct TradeLog {
    trades: Vec<TradeRecord>,
}

pub struct TradeRecord {
    pub signal: TradeSignal,
    pub amount_usd: f64,
    pub tx_hash: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl TradeLog {
    pub fn new() -> Self {
        Self { trades: Vec::new() }
    }

    pub fn record(&mut self, signal: TradeSignal, amount_usd: f64, tx_hash: String) {
        self.trades.push(TradeRecord {
            signal,
            amount_usd,
            tx_hash,
            timestamp: chrono::Utc::now(),
        });
    }

    pub fn summary(&self) -> HashMap<String, f64> {
        let mut exposure: HashMap<String, f64> = HashMap::new();
        for trade in &self.trades {
            *exposure.entry(trade.signal.market_id.clone()).or_default() += trade.amount_usd;
        }
        exposure
    }
}
