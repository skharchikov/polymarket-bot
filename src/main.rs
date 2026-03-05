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
use data::crawler::Crawler;

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

    // 1. Crawl large dataset — 2000 markets, crypto only
    let crawler = Crawler::new(100);
    let dataset = crawler.build_dataset(2000, true).await?;

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
        edge_threshold: 0.05,
        position_size_pct: 0.02,
    };

    // Strategy 1: SMA Momentum
    let sma = run_backtest(&dataset, &config, |market, price| {
        let h = &market.price_history;
        if h.len() < 10 {
            return (price, 0.3);
        }
        let recent: Vec<f64> = h.iter().rev().take(10).map(|t| t.p).collect();
        let sma3: f64 = recent[..3].iter().sum::<f64>() / 3.0;
        let sma10: f64 = recent.iter().sum::<f64>() / 10.0;
        let momentum = sma3 - sma10;
        let est = (price + momentum * 2.0).clamp(0.05, 0.95);
        let conf = (momentum.abs() * 10.0).clamp(0.2, 0.8);
        (est, conf)
    });
    log_results("SMA Momentum (3/10)", &config, &sma);

    // Strategy 2: Mean Reversion
    let mr = run_backtest(&dataset, &config, |market, price| {
        let h = &market.price_history;
        if h.len() < 20 {
            return (price, 0.2);
        }
        let avg: f64 = h.iter().map(|t| t.p).sum::<f64>() / h.len() as f64;
        let deviation = price - avg;
        let est = (price - deviation * 0.4).clamp(0.05, 0.95);
        let conf = (deviation.abs() * 5.0).clamp(0.1, 0.7);
        (est, conf)
    });
    log_results("Mean Reversion", &config, &mr);

    // Strategy 3: Trend Following with volatility filter
    let tf = run_backtest(&dataset, &config, |market, price| {
        let h = &market.price_history;
        if h.len() < 20 {
            return (price, 0.2);
        }
        let prices: Vec<f64> = h.iter().map(|t| t.p).collect();
        let n = prices.len();

        let first_half: f64 = prices[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
        let second_half: f64 = prices[n / 2..].iter().sum::<f64>() / (n - n / 2) as f64;
        let trend = second_half - first_half;

        let recent: Vec<f64> = prices.iter().rev().take(10).copied().collect();
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let vol: f64 =
            (recent.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / recent.len() as f64).sqrt();

        if vol > 0.15 || vol < 0.01 {
            return (price, 0.1);
        }

        let est = (price + trend * 1.5).clamp(0.05, 0.95);
        let conf = (trend.abs() * 8.0).clamp(0.2, 0.85);
        (est, conf)
    });
    log_results("Trend Following", &config, &tf);

    // Strategy 4: Contrarian
    let contrarian = run_backtest(&dataset, &config, |_market, price| {
        let distance_from_half = (price - 0.5).abs();
        if distance_from_half < 0.15 {
            return (price, 0.1);
        }
        let est = price + (0.5 - price) * 0.3;
        let conf = (distance_from_half * 2.0).clamp(0.3, 0.8);
        (est, conf)
    });
    log_results("Contrarian", &config, &contrarian);

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
