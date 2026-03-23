use std::collections::HashMap;

use super::metrics::BacktestMetrics;
use super::portfolio::{Portfolio, Side};
use crate::data::models::{HistoricalMarket, PriceTick};

pub struct BacktestConfig {
    pub starting_cash: f64,
    pub edge_threshold: f64,
    pub position_size_pct: f64,
    /// Slippage as a fraction of price (e.g., 0.01 = 1%)
    pub slippage_pct: f64,
    /// Trading fee as a fraction of notional (e.g., 0.02 = 2%)
    pub fee_pct: f64,
    /// How far into the history to enter (fraction 0.0-1.0).
    /// 0.2 means enter after observing the first 20% of ticks.
    pub entry_point: f64,
    /// Minimum observed ticks before entry (overrides entry_point if too few).
    pub min_lookback: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            starting_cash: 10_000.0,
            edge_threshold: 0.03,
            position_size_pct: 0.05,
            slippage_pct: 0.01,
            fee_pct: 0.02,
            entry_point: 0.2,
            min_lookback: 10,
        }
    }
}

pub struct BacktestResult {
    pub portfolio: Portfolio,
    pub metrics: BacktestMetrics,
}

/// Estimate function receives (observed_history, current_price) and returns (probability, confidence).
/// observed_history contains ONLY the ticks available at entry time — no future data.
pub fn run_backtest<F>(
    markets: &[HistoricalMarket],
    config: &BacktestConfig,
    mut estimate_fn: F,
) -> BacktestResult
where
    F: FnMut(&[PriceTick], f64) -> (f64, f64),
{
    let mut portfolio =
        Portfolio::with_costs(config.starting_cash, config.slippage_pct, config.fee_pct);
    let mut predictions: Vec<(f64, bool)> = Vec::new();

    for market in markets {
        let n = market.price_history.len();
        let entry_idx = ((n as f64 * config.entry_point) as usize).max(config.min_lookback);

        if entry_idx >= n {
            continue;
        }

        let observed = &market.price_history[..entry_idx];
        let entry_price = market.price_history[entry_idx].p;

        if entry_price <= 0.02 || entry_price >= 0.98 {
            continue;
        }

        let (estimated_prob, confidence) = estimate_fn(observed, entry_price);
        predictions.push((estimated_prob, market.resolved_yes));

        let edge = estimated_prob - entry_price;
        let effective_edge = edge * confidence;

        if effective_edge.abs() < config.edge_threshold {
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
        // 20 ticks, entry at tick 4 (20% of 20), oracle knows outcome
        let mut prices_yes = vec![0.50; 20];
        prices_yes[18] = 0.55;
        prices_yes[19] = 0.60;
        let mut prices_no = vec![0.80; 20];
        prices_no[18] = 0.75;
        prices_no[19] = 0.70;

        let markets = vec![
            make_market("m1", prices_yes, true),
            make_market("m2", prices_no, false),
        ];

        let config = BacktestConfig {
            starting_cash: 1000.0,
            edge_threshold: 0.05,
            position_size_pct: 0.10,
            slippage_pct: 0.0,
            fee_pct: 0.0,
            entry_point: 0.2,
            min_lookback: 3,
        };

        let result = run_backtest(&markets, &config, |observed, _price| {
            // Oracle: if observed prices are around 0.50, probably YES; around 0.80, probably NO
            let avg: f64 = observed.iter().map(|t| t.p).sum::<f64>() / observed.len() as f64;
            if avg < 0.6 { (0.90, 1.0) } else { (0.10, 1.0) }
        });

        assert!(result.metrics.roi > 0.0, "Should be profitable with oracle");
        assert_eq!(result.metrics.total_trades, 2);
        assert_eq!(result.metrics.winning_trades, 2);
    }

    #[test]
    fn test_backtest_no_edge_no_trades() {
        let markets = vec![make_market("m1", vec![0.50; 20], true)];

        let config = BacktestConfig {
            starting_cash: 1000.0,
            edge_threshold: 0.05,
            position_size_pct: 0.10,
            slippage_pct: 0.0,
            fee_pct: 0.0,
            entry_point: 0.2,
            min_lookback: 3,
        };

        let result = run_backtest(&markets, &config, |_, price| (price, 1.0));

        assert_eq!(result.metrics.total_trades, 0);
    }

    #[test]
    fn test_slippage_and_fees_reduce_profit() {
        let mut prices = vec![0.50; 20];
        prices[18] = 0.55;
        prices[19] = 0.60;
        let markets = vec![make_market("m1", prices, true)];

        let no_costs = BacktestConfig {
            starting_cash: 1000.0,
            edge_threshold: 0.05,
            position_size_pct: 0.10,
            slippage_pct: 0.0,
            fee_pct: 0.0,
            entry_point: 0.2,
            min_lookback: 3,
        };

        let with_costs = BacktestConfig {
            slippage_pct: 0.02,
            fee_pct: 0.02,
            ..no_costs
        };

        let r1 = run_backtest(&markets, &no_costs, |_, _| (0.90, 1.0));
        let r2 = run_backtest(&markets, &with_costs, |_, _| (0.90, 1.0));

        assert!(
            r2.metrics.total_pnl < r1.metrics.total_pnl,
            "Costs should reduce profit"
        );
    }

    #[test]
    fn test_skips_markets_with_too_few_ticks() {
        let markets = vec![make_market("m1", vec![0.50; 5], true)];

        let config = BacktestConfig {
            min_lookback: 10,
            ..Default::default()
        };

        let result = run_backtest(&markets, &config, |_, _| (0.90, 1.0));
        assert_eq!(result.metrics.total_trades, 0);
    }
}
