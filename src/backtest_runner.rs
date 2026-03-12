use anyhow::Result;

use crate::backtest::engine::{BacktestConfig, BacktestResult, run_backtest};
use crate::data::crawler::{Crawler, CrawlerConfig};

pub async fn run() -> Result<()> {
    tracing::info!("Polymarket backtest starting...");

    let crawler = Crawler::new(CrawlerConfig {
        market_limit: 2000,
        crypto_only: true,
        min_volume: 500.0,
        min_ticks: 20,
        min_duration_hours: 4.0,
        ..CrawlerConfig::default()
    });
    let dataset = crawler.build_dataset().await?;

    if dataset.is_empty() {
        tracing::warn!("No historical markets found");
        return Ok(());
    }

    let yes_count = dataset.iter().filter(|m| m.resolved_yes).count();
    tracing::info!(
        markets = dataset.len(),
        yes = yes_count,
        no = dataset.len() - yes_count,
        avg_ticks = dataset.iter().map(|m| m.price_history.len()).sum::<usize>() / dataset.len(),
        "Dataset ready"
    );

    let config = BacktestConfig {
        starting_cash: 300.0,
        edge_threshold: 0.03,
        position_size_pct: 0.05,
        slippage_pct: 0.01,
        fee_pct: 0.02,
        entry_point: 0.2,
        min_lookback: 10,
    };

    // Strategy 1: SMA Crossover
    let sma = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 10 {
            return (price, 0.1);
        }
        let prices: Vec<f64> = observed.iter().map(|t| t.p).collect();
        let short_w = (n / 3).max(3);
        let sma_short: f64 = prices[n - short_w..].iter().sum::<f64>() / short_w as f64;
        let sma_long: f64 = prices.iter().sum::<f64>() / n as f64;
        let momentum = sma_short - sma_long;
        let est = (price + momentum * 3.0).clamp(0.05, 0.95);
        let conf = (momentum.abs() * 8.0).clamp(0.2, 0.9);
        (est, conf)
    });
    log_result("SMA Crossover", &config, &sma);

    // Strategy 2: Mean Reversion
    let mr = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 5 {
            return (price, 0.1);
        }
        let prices: Vec<f64> = observed.iter().map(|t| t.p).collect();
        let avg: f64 = prices.iter().sum::<f64>() / n as f64;
        let std_dev: f64 =
            (prices.iter().map(|p| (p - avg).powi(2)).sum::<f64>() / n as f64).sqrt();
        let deviation = price - avg;
        if std_dev < 0.01 || deviation.abs() < std_dev {
            return (price, 0.1);
        }
        let est = (price - deviation * 0.5).clamp(0.05, 0.95);
        let conf = ((deviation.abs() / std_dev) * 0.2).clamp(0.2, 0.8);
        (est, conf)
    });
    log_result("Mean Reversion", &config, &mr);

    // Strategy 3: Trend Following
    let tf = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 10 {
            return (price, 0.1);
        }
        let prices: Vec<f64> = observed.iter().map(|t| t.p).collect();
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean: f64 = prices.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &p) in prices.iter().enumerate() {
            let x = i as f64 - x_mean;
            num += x * (p - y_mean);
            den += x * x;
        }
        let slope = if den > 0.0 { num / den } else { 0.0 };
        let recent = &prices[n.saturating_sub(5)..];
        let rmean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let vol: f64 =
            (recent.iter().map(|p| (p - rmean).powi(2)).sum::<f64>() / recent.len() as f64).sqrt();
        if vol > 0.12 || slope.abs() < 0.001 {
            return (price, 0.1);
        }
        let est = (price + slope * n as f64 * 0.5).clamp(0.05, 0.95);
        let conf = (slope.abs() * n as f64 * 3.0).clamp(0.2, 0.85);
        (est, conf)
    });
    log_result("Trend Following", &config, &tf);

    // Strategy 4: Contrarian at extremes
    let ct = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 5 {
            return (price, 0.1);
        }
        let distance = (price - 0.5).abs();
        if distance < 0.20 {
            return (price, 0.1);
        }
        let recent: Vec<f64> = observed.iter().rev().take(5).map(|t| t.p).collect();
        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let accel = (price - recent_avg).abs();
        if accel < 0.02 {
            return (price, 0.1);
        }
        let est = price + (0.5 - price) * 0.25;
        let conf = (distance * 1.5 + accel * 3.0).clamp(0.3, 0.85);
        (est, conf)
    });
    log_result("Contrarian Extremes", &config, &ct);

    Ok(())
}

fn log_result(name: &str, config: &BacktestConfig, result: &BacktestResult) {
    let m = &result.metrics;
    tracing::info!(
        strategy = name,
        start = format_args!("€{:.0}", config.starting_cash),
        end = format_args!("€{:.0}", result.portfolio.total_equity()),
        roi = format_args!("{:.2}%", m.roi * 100.0),
        trades = m.total_trades,
        wins = m.winning_trades,
        losses = m.total_trades - m.winning_trades,
        win_rate = format_args!("{:.1}%", m.win_rate * 100.0),
        pnl = format_args!("€{:.2}", m.total_pnl),
        max_dd = format_args!("{:.2}%", m.max_drawdown * 100.0),
        sharpe = format_args!("{:.3}", m.sharpe_ratio),
        brier = format_args!("{:.4}", m.brier_score),
        "Backtest result"
    );
}
