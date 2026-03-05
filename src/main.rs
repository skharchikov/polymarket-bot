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
use backtest::engine::{BacktestConfig, run_backtest};
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

    tracing::info!("Polymarket trading bot starting...");

    // 1. Crawl historical data
    let crawler = Crawler::new(200); // 200ms between requests
    tracing::info!("Fetching resolved markets and price history...");
    let dataset = crawler.build_dataset(200, true).await?;

    if dataset.is_empty() {
        tracing::warn!("No historical markets with price data found");
        return Ok(());
    }

    tracing::info!(
        markets = dataset.len(),
        "Dataset ready, running backtest..."
    );

    // 2. Run backtest with a simple mean-reversion estimator
    let config = BacktestConfig {
        starting_cash: 10_000.0,
        edge_threshold: 0.05,
        position_size_pct: 0.02,
    };

    let result = run_backtest(&dataset, &config, |market, current_price| {
        // Simple contrarian estimator: if price is extreme, bet on reversion
        // In reality this would be an LLM call or a proper model
        let history = &market.price_history;
        if history.len() < 5 {
            return current_price; // no edge
        }

        // Average of last 5 prices as a simple moving average
        let recent_avg: f64 = history.iter().rev().take(5).map(|t| t.p).sum::<f64>() / 5.0;

        // Blend: 70% market price, 30% recent trend
        0.7 * current_price + 0.3 * recent_avg
    });

    // 3. Print results
    let m = &result.metrics;
    println!("\n========== BACKTEST RESULTS ==========");
    println!("Starting capital:  ${:.2}", config.starting_cash);
    println!("Final equity:      ${:.2}", result.portfolio.total_equity());
    println!("ROI:               {:.2}%", m.roi * 100.0);
    println!("Total trades:      {}", m.total_trades);
    println!("Winning trades:    {}", m.winning_trades);
    println!("Win rate:          {:.1}%", m.win_rate * 100.0);
    println!("Total PnL:         ${:.2}", m.total_pnl);
    println!("Max drawdown:      {:.2}%", m.max_drawdown * 100.0);
    println!("Sharpe ratio:      {:.3}", m.sharpe_ratio);
    println!(
        "Brier score:       {:.4} (lower is better, 0.25 = random)",
        m.brier_score
    );
    println!("=======================================\n");

    Ok(())
}
