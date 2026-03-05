use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Yes,
    No,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub market_id: String,
    pub side: Side,
    pub size: f64,
    pub entry_price: f64,
}

#[derive(Debug)]
pub struct Portfolio {
    pub starting_cash: f64,
    pub cash: f64,
    pub positions: HashMap<String, Position>,
    pub equity_curve: Vec<f64>,
    pub trade_results: Vec<TradeResult>,
    pub slippage_pct: f64,
    pub fee_pct: f64,
}

#[derive(Debug, Clone)]
pub struct TradeResult {
    pub market_id: String,
    pub side: Side,
    pub entry_price: f64,
    pub size: f64,
    pub pnl: f64,
}

impl Portfolio {
    pub fn new(starting_cash: f64) -> Self {
        Self::with_costs(starting_cash, 0.0, 0.0)
    }

    pub fn with_costs(starting_cash: f64, slippage_pct: f64, fee_pct: f64) -> Self {
        Self {
            starting_cash,
            cash: starting_cash,
            positions: HashMap::new(),
            equity_curve: vec![starting_cash],
            trade_results: Vec::new(),
            slippage_pct,
            fee_pct,
        }
    }

    /// Open a position. Cost = size * effective_price (with slippage + fees).
    pub fn open_position(&mut self, market_id: &str, side: Side, size: f64, price: f64) -> bool {
        // Slippage moves the price against us
        let slipped_price = (price * (1.0 + self.slippage_pct)).min(0.99);
        let fee = size * slipped_price * self.fee_pct;
        let cost = size * slipped_price + fee;

        if cost > self.cash {
            return false;
        }

        self.cash -= cost;
        self.positions.insert(
            market_id.to_string(),
            Position {
                market_id: market_id.to_string(),
                side,
                size,
                entry_price: slipped_price,
            },
        );
        true
    }

    /// Resolve a position given the market outcome.
    pub fn resolve(&mut self, market_id: &str, resolved_yes: bool) -> Option<f64> {
        let pos = self.positions.remove(market_id)?;
        let won =
            (pos.side == Side::Yes && resolved_yes) || (pos.side == Side::No && !resolved_yes);

        let payout = if won { pos.size } else { 0.0 };
        // Fee on payout
        let exit_fee = payout * self.fee_pct;
        let net_payout = payout - exit_fee;
        let cost = pos.size * pos.entry_price;
        let pnl = net_payout - cost;

        self.cash += net_payout;
        self.trade_results.push(TradeResult {
            market_id: market_id.to_string(),
            side: pos.side,
            entry_price: pos.entry_price,
            size: pos.size,
            pnl,
        });

        Some(pnl)
    }

    /// Record current equity (cash + mark-to-market positions).
    pub fn snapshot_equity(&mut self, current_prices: &HashMap<String, f64>) {
        let positions_value: f64 = self
            .positions
            .iter()
            .map(|(id, pos)| {
                let price = current_prices.get(id).copied().unwrap_or(pos.entry_price);
                pos.size * price
            })
            .sum();
        self.equity_curve.push(self.cash + positions_value);
    }

    pub fn total_equity(&self) -> f64 {
        *self.equity_curve.last().unwrap_or(&self.cash)
    }
}
