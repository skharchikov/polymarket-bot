use super::portfolio::Portfolio;

#[derive(Debug)]
pub struct BacktestMetrics {
    pub roi: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub brier_score: f64,
}

impl BacktestMetrics {
    pub fn compute(portfolio: &Portfolio, predictions: &[(f64, bool)]) -> Self {
        let roi = (portfolio.total_equity() - portfolio.starting_cash) / portfolio.starting_cash;
        let total_trades = portfolio.trade_results.len();
        let winning_trades = portfolio
            .trade_results
            .iter()
            .filter(|t| t.pnl > 0.0)
            .count();
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };
        let total_pnl: f64 = portfolio.trade_results.iter().map(|t| t.pnl).sum();
        let max_drawdown = compute_max_drawdown(&portfolio.equity_curve);
        let sharpe_ratio = compute_sharpe(&portfolio.equity_curve);
        let brier_score = compute_brier_score(predictions);

        Self {
            roi,
            total_trades,
            winning_trades,
            win_rate,
            total_pnl,
            max_drawdown,
            sharpe_ratio,
            brier_score,
        }
    }
}

fn compute_max_drawdown(equity_curve: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;

    for &equity in equity_curve {
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

fn compute_sharpe(equity_curve: &[f64]) -> f64 {
    if equity_curve.len() < 2 {
        return 0.0;
    }

    let returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    mean / std_dev
}

/// Brier score: average of (predicted_probability - actual_outcome)^2.
/// Lower is better. 0 = perfect, 0.25 = random.
fn compute_brier_score(predictions: &[(f64, bool)]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let sum: f64 = predictions
        .iter()
        .map(|(prob, outcome)| {
            let actual = if *outcome { 1.0 } else { 0.0 };
            (prob - actual).powi(2)
        })
        .sum();
    sum / predictions.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown() {
        let curve = vec![100.0, 110.0, 90.0, 95.0, 80.0, 120.0];
        let dd = compute_max_drawdown(&curve);
        // Peak was 110, lowest after was 80 => dd = 30/110 ≈ 0.2727
        assert!((dd - 30.0 / 110.0).abs() < 1e-6);
    }

    #[test]
    fn test_brier_score_perfect() {
        let preds = vec![(1.0, true), (0.0, false)];
        assert!((compute_brier_score(&preds)).abs() < 1e-10);
    }

    #[test]
    fn test_brier_score_worst() {
        let preds = vec![(0.0, true), (1.0, false)];
        assert!((compute_brier_score(&preds) - 1.0).abs() < 1e-10);
    }
}
