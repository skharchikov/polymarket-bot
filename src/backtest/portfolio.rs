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
        Self {
            starting_cash,
            cash: starting_cash,
            positions: HashMap::new(),
            equity_curve: vec![starting_cash],
            trade_results: Vec::new(),
        }
    }

    /// Open a position. Cost = size * price.
    pub fn open_position(&mut self, market_id: &str, side: Side, size: f64, price: f64) -> bool {
        let cost = size * price;
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
                entry_price: price,
            },
        );
        true
    }

    /// Resolve a position given the market outcome.
    /// Returns PnL for the trade.
    pub fn resolve(&mut self, market_id: &str, resolved_yes: bool) -> Option<f64> {
        let pos = self.positions.remove(market_id)?;
        let won =
            (pos.side == Side::Yes && resolved_yes) || (pos.side == Side::No && !resolved_yes);

        let payout = if won { pos.size } else { 0.0 };
        let cost = pos.size * pos.entry_price;
        let pnl = payout - cost;

        self.cash += payout;
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
