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
use backtest::engine::{BacktestConfig, BacktestResult, run_backtest, run_backtest_async};
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

    // 1. Crawl historical crypto markets
    let crawler = Crawler::new(200);
    tracing::info!("Fetching resolved crypto markets...");
    let dataset = crawler.build_dataset(200, true).await?;

    if dataset.is_empty() {
        tracing::warn!("No historical markets with price data found");
        return Ok(());
    }

    tracing::info!(markets = dataset.len(), "Dataset ready");

    let config = BacktestConfig {
        starting_cash: 10_000.0,
        edge_threshold: 0.05,
        position_size_pct: 0.02,
    };

    // 2. Baseline: SMA momentum estimator
    tracing::info!("Running SMA baseline backtest...");
    let sma_result = run_backtest(&dataset, &config, |market, current_price| {
        let history = &market.price_history;
        if history.len() < 10 {
            return (current_price, 0.3);
        }

        let recent: Vec<f64> = history.iter().rev().take(10).map(|t| t.p).collect();
        let sma_short: f64 = recent[..3].iter().sum::<f64>() / 3.0;
        let sma_long: f64 = recent.iter().sum::<f64>() / 10.0;

        // Momentum: if short SMA > long SMA, trend is up
        let momentum = sma_short - sma_long;
        let estimate = (current_price + momentum * 2.0).clamp(0.05, 0.95);
        let confidence = (momentum.abs() * 10.0).clamp(0.2, 0.8);

        (estimate, confidence)
    });

    print_results("SMA Momentum", &config, &sma_result);

    // 3. LLM-powered estimator (if OPENAI_API_KEY is set)
    if std::env::var("OPENAI_API_KEY").is_ok() {
        tracing::info!("Running LLM-powered backtest...");
        let estimator = agent::estimator::LlmEstimator::new();

        // Limit LLM calls to save API costs — use a subset
        let llm_subset: Vec<_> = dataset.iter().take(15).cloned().collect();

        let llm_result = run_backtest_async(&llm_subset, &config, |market, price| {
            let question = market.question.clone();
            let est = &estimator;
            async move {
                match est.estimate(&question, price).await {
                    Ok((prob, conf)) => {
                        tracing::info!(
                            question = %question,
                            market_price = price,
                            llm_prob = prob,
                            confidence = conf,
                            "LLM estimate"
                        );
                        (prob, conf)
                    }
                    Err(e) => {
                        tracing::warn!(err = %e, "LLM estimate failed, using market price");
                        (price, 0.0)
                    }
                }
            }
        })
        .await;

        print_results("LLM (GPT-4o-mini)", &config, &llm_result);
    } else {
        tracing::warn!("OPENAI_API_KEY not set, skipping LLM backtest");
    }

    Ok(())
}

fn print_results(name: &str, config: &BacktestConfig, result: &BacktestResult) {
    let m = &result.metrics;
    println!("\n===== {name} BACKTEST =====");
    println!("Starting capital:  ${:.2}", config.starting_cash);
    println!("Final equity:      ${:.2}", result.portfolio.total_equity());
    println!("ROI:               {:.2}%", m.roi * 100.0);
    println!("Total trades:      {}", m.total_trades);
    println!("Winning trades:    {}", m.winning_trades);
    println!(
        "Win rate:          {:.1}%",
        if m.total_trades > 0 {
            m.win_rate * 100.0
        } else {
            0.0
        }
    );
    println!("Total PnL:         ${:.2}", m.total_pnl);
    println!("Max drawdown:      {:.2}%", m.max_drawdown * 100.0);
    println!("Sharpe ratio:      {:.3}", m.sharpe_ratio);
    println!(
        "Brier score:       {:.4} (0=perfect, 0.25=random)",
        m.brier_score
    );
    println!("===========================\n");
}
