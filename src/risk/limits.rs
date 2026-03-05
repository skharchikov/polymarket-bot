use std::collections::HashMap;

pub struct RiskLimits {
    pub max_per_market_pct: f64,
    pub max_total_exposure_pct: f64,
    pub max_daily_trades: usize,
    positions: HashMap<String, f64>,
    daily_trade_count: usize,
}

impl RiskLimits {
    pub fn new(max_per_market_pct: f64, max_total_exposure_pct: f64, max_daily_trades: usize) -> Self {
        Self {
            max_per_market_pct,
            max_total_exposure_pct,
            max_daily_trades,
            positions: HashMap::new(),
            daily_trade_count: 0,
        }
    }

    pub fn can_trade(&self, market_id: &str, amount_pct: f64, bankroll: f64) -> bool {
        if self.daily_trade_count >= self.max_daily_trades {
            tracing::warn!("Daily trade limit reached");
            return false;
        }

        let current = self.positions.get(market_id).copied().unwrap_or(0.0);
        if (current + amount_pct * bankroll) / bankroll > self.max_per_market_pct {
            tracing::warn!(market_id, "Per-market exposure limit reached");
            return false;
        }

        let total: f64 = self.positions.values().sum();
        if (total + amount_pct * bankroll) / bankroll > self.max_total_exposure_pct {
            tracing::warn!("Total exposure limit reached");
            return false;
        }

        true
    }

    pub fn record_trade(&mut self, market_id: &str, amount: f64) {
        *self.positions.entry(market_id.to_string()).or_default() += amount;
        self.daily_trade_count += 1;
    }

    pub fn reset_daily(&mut self) {
        self.daily_trade_count = 0;
    }
}
