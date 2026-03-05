mod agent;
mod backtest;
mod data;
mod execution;
mod markets;
mod pricing;
mod risk;
mod storage;
mod strategies;

use anyhow::Result;
use backtest::engine::{BacktestConfig, BacktestResult, run_backtest};
use data::crawler::{Crawler, CrawlerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("polymarket_bot=info".parse()?),
        )
        .init();

    dotenvy::dotenv().ok();

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
    let no_count = dataset.len() - yes_count;
    let avg_ticks: f64 = dataset
        .iter()
        .map(|m| m.price_history.len() as f64)
        .sum::<f64>()
        / dataset.len() as f64;

    tracing::info!(
        markets = dataset.len(),
        yes = yes_count,
        no = no_count,
        avg_ticks = avg_ticks as usize,
        "Dataset ready"
    );

    let config = BacktestConfig {
        starting_cash: 10_000.0,
        edge_threshold: 0.03,
        position_size_pct: 0.05,
        slippage_pct: 0.01,
        fee_pct: 0.02,
        entry_point: 0.2,
        min_lookback: 10,
    };

    // Strategy 1: SMA Crossover — short SMA vs long SMA on observed history
    let sma = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 10 {
            return (price, 0.1);
        }
        let prices: Vec<f64> = observed.iter().map(|t| t.p).collect();
        let short_window = (n / 3).max(3);
        let sma_short: f64 = prices[n - short_window..].iter().sum::<f64>() / short_window as f64;
        let sma_long: f64 = prices.iter().sum::<f64>() / n as f64;
        let momentum = sma_short - sma_long;

        let est = (price + momentum * 3.0).clamp(0.05, 0.95);
        let conf = (momentum.abs() * 8.0).clamp(0.2, 0.9);
        (est, conf)
    });
    log_results("SMA Crossover", &config, &sma);

    // Strategy 2: Mean Reversion — price deviates from observed average
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
        // Only trade when price has moved significantly from mean
        if std_dev < 0.01 || deviation.abs() < std_dev {
            return (price, 0.1);
        }

        // Expect reversion toward mean
        let est = (price - deviation * 0.5).clamp(0.05, 0.95);
        let conf = ((deviation.abs() / std_dev) * 0.2).clamp(0.2, 0.8);
        (est, conf)
    });
    log_results("Mean Reversion", &config, &mr);

    // Strategy 3: Trend Following with volatility filter
    let tf = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 10 {
            return (price, 0.1);
        }
        let prices: Vec<f64> = observed.iter().map(|t| t.p).collect();

        // Compute trend: linear regression slope
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

        // Volatility filter
        let recent = &prices[n.saturating_sub(5)..];
        let recent_mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let vol: f64 = (recent
            .iter()
            .map(|p| (p - recent_mean).powi(2))
            .sum::<f64>()
            / recent.len() as f64)
            .sqrt();

        // Skip high-volatility or flat markets
        if vol > 0.12 || slope.abs() < 0.001 {
            return (price, 0.1);
        }

        // Project trend forward
        let est = (price + slope * n as f64 * 0.5).clamp(0.05, 0.95);
        let conf = (slope.abs() * n as f64 * 3.0).clamp(0.2, 0.85);
        (est, conf)
    });
    log_results("Trend Following", &config, &tf);

    // Strategy 4: Contrarian at extremes
    let contrarian = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 5 {
            return (price, 0.1);
        }

        let distance = (price - 0.5).abs();
        if distance < 0.20 {
            return (price, 0.1); // Only trade extremes
        }

        // Check if trend is accelerating toward extreme (overreaction)
        let recent: Vec<f64> = observed.iter().rev().take(5).map(|t| t.p).collect();
        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let acceleration = (price - recent_avg).abs();

        if acceleration < 0.02 {
            return (price, 0.1); // Only bet against rapid moves
        }

        // Bet on reversion from extreme
        let est = price + (0.5 - price) * 0.25;
        let conf = (distance * 1.5 + acceleration * 3.0).clamp(0.3, 0.85);
        (est, conf)
    });
    log_results("Contrarian Extremes", &config, &contrarian);

    // Strategy 5: Combined — blend SMA + mean reversion signals
    let combined = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 10 {
            return (price, 0.1);
        }
        let prices: Vec<f64> = observed.iter().map(|t| t.p).collect();

        // SMA signal
        let short_w = (n / 3).max(3);
        let sma_short: f64 = prices[n - short_w..].iter().sum::<f64>() / short_w as f64;
        let sma_long: f64 = prices.iter().sum::<f64>() / n as f64;
        let sma_signal = sma_short - sma_long;

        // Mean reversion signal
        let deviation = price - sma_long;
        let std_dev: f64 =
            (prices.iter().map(|p| (p - sma_long).powi(2)).sum::<f64>() / n as f64).sqrt();
        let mr_signal = if std_dev > 0.01 {
            -deviation / std_dev * 0.05
        } else {
            0.0
        };

        // Blend: trend when vol is low, mean-revert when price is extreme
        let distance = (price - 0.5).abs();
        let trend_weight = if distance > 0.25 { 0.3 } else { 0.7 };
        let blended = sma_signal * trend_weight + mr_signal * (1.0 - trend_weight);

        let est = (price + blended * 2.0).clamp(0.05, 0.95);
        let conf = (blended.abs() * 10.0).clamp(0.15, 0.8);
        (est, conf)
    });
    log_results("Combined SMA+MR", &config, &combined);

    Ok(())
}

fn log_results(name: &str, config: &BacktestConfig, result: &BacktestResult) {
    let m = &result.metrics;
    tracing::info!(
        strategy = name,
        start = format_args!("${:.0}", config.starting_cash),
        end = format_args!("${:.0}", result.portfolio.total_equity()),
        roi = format_args!("{:.2}%", m.roi * 100.0),
        trades = m.total_trades,
        wins = m.winning_trades,
        losses = m.total_trades - m.winning_trades,
        win_rate = format_args!("{:.1}%", m.win_rate * 100.0),
        pnl = format_args!("${:.2}", m.total_pnl),
        max_dd = format_args!("{:.2}%", m.max_drawdown * 100.0),
        sharpe = format_args!("{:.3}", m.sharpe_ratio),
        brier = format_args!("{:.4}", m.brier_score),
        "Backtest result"
    );
}
