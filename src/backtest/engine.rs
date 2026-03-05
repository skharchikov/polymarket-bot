use std::collections::HashMap;

use super::metrics::BacktestMetrics;
use super::portfolio::{Portfolio, Side};
use crate::data::models::HistoricalMarket;

pub struct BacktestConfig {
    pub starting_cash: f64,
    pub edge_threshold: f64,
    pub position_size_pct: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            starting_cash: 10_000.0,
            edge_threshold: 0.05,
            position_size_pct: 0.02,
        }
    }
}

pub struct BacktestResult {
    pub portfolio: Portfolio,
    pub metrics: BacktestMetrics,
}

/// Simple value strategy backtester.
///
/// For each historical market:
/// 1. Walk through price ticks
/// 2. Use a probability estimator to get an estimate
/// 3. If edge > threshold, open a position at current price
/// 4. Resolve at market end
pub fn run_backtest<F>(
    markets: &[HistoricalMarket],
    config: &BacktestConfig,
    mut estimate_fn: F,
) -> BacktestResult
where
    F: FnMut(&HistoricalMarket, f64) -> f64,
{
    let mut portfolio = Portfolio::new(config.starting_cash);
    let mut predictions: Vec<(f64, bool)> = Vec::new();

    for market in markets {
        if market.price_history.is_empty() {
            continue;
        }

        // Use the midpoint price as entry signal
        let mid_idx = market.price_history.len() / 2;
        let entry_price = market.price_history[mid_idx].p;

        if entry_price <= 0.0 || entry_price >= 1.0 {
            continue;
        }

        let estimated_prob = estimate_fn(market, entry_price);
        predictions.push((estimated_prob, market.resolved_yes));

        let edge = estimated_prob - entry_price;

        if edge.abs() < config.edge_threshold {
            continue;
        }

        let (side, price) = if edge > 0.0 {
            (Side::Yes, entry_price)
        } else {
            (Side::No, 1.0 - entry_price)
        };

        let size = config.position_size_pct * portfolio.cash / price;
        if size <= 0.0 {
            continue;
        }

        if portfolio.open_position(&market.market_id, side, size, price) {
            // Snapshot equity at entry
            let prices: HashMap<String, f64> =
                HashMap::from([(market.market_id.clone(), entry_price)]);
            portfolio.snapshot_equity(&prices);
        }
    }

    // Resolve all positions
    let market_map: HashMap<&str, bool> = markets
        .iter()
        .map(|m| (m.market_id.as_str(), m.resolved_yes))
        .collect();

    let open_ids: Vec<String> = portfolio.positions.keys().cloned().collect();
    for market_id in open_ids {
        if let Some(&resolved_yes) = market_map.get(market_id.as_str()) {
            portfolio.resolve(&market_id, resolved_yes);
        }
    }

    portfolio.snapshot_equity(&HashMap::new());

    let metrics = BacktestMetrics::compute(&portfolio, &predictions);
    BacktestResult { portfolio, metrics }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::models::{HistoricalMarket, PriceTick};
    use chrono::Utc;

    fn make_market(id: &str, prices: Vec<f64>, resolved_yes: bool) -> HistoricalMarket {
        HistoricalMarket {
            market_id: id.to_string(),
            question: format!("Test market {id}"),
            token_id: format!("token_{id}"),
            end_date: Utc::now(),
            resolved_yes,
            price_history: prices
                .into_iter()
                .enumerate()
                .map(|(i, p)| PriceTick {
                    t: i as i64 * 3600,
                    p,
                })
                .collect(),
        }
    }

    #[test]
    fn test_backtest_profitable_edge() {
        let markets = vec![
            // Market priced at 0.50 but resolves YES - if we estimate 0.80 we should profit
            make_market("m1", vec![0.50, 0.50, 0.50, 0.55, 0.60], true),
            // Market priced at 0.80 but resolves NO - if we estimate 0.30 we should profit
            make_market("m2", vec![0.80, 0.80, 0.80, 0.75, 0.70], false),
        ];

        let config = BacktestConfig {
            starting_cash: 1000.0,
            edge_threshold: 0.05,
            position_size_pct: 0.10,
        };

        // Oracle estimator: knows the true outcome
        let result = run_backtest(&markets, &config, |market, _price| {
            if market.resolved_yes { 0.90 } else { 0.10 }
        });

        assert!(result.metrics.roi > 0.0, "Should be profitable with oracle");
        assert_eq!(result.metrics.total_trades, 2);
        assert_eq!(result.metrics.winning_trades, 2);
    }

    #[test]
    fn test_backtest_no_edge_no_trades() {
        let markets = vec![make_market("m1", vec![0.50, 0.50, 0.50], true)];

        let config = BacktestConfig {
            starting_cash: 1000.0,
            edge_threshold: 0.05,
            position_size_pct: 0.10,
        };

        // Estimator agrees with market - no edge
        let result = run_backtest(&markets, &config, |_market, price| price);

        assert_eq!(result.metrics.total_trades, 0);
        assert!((result.metrics.roi).abs() < 1e-10);
    }
}
