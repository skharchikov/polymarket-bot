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
use markets::fetcher::MarketFetcher;
use pricing::probability::ProbabilityEstimate;
use strategies::mispricing::{self, StrategyConfig};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    dotenvy::dotenv().ok();

    tracing::info!("Polymarket trading bot starting...");

    let fetcher = MarketFetcher::new();
    let config = StrategyConfig::default();
    let bankroll = 1000.0;

    // Fetch active markets
    let markets = fetcher.fetch_markets(10).await?;
    tracing::info!(count = markets.len(), "Fetched markets");

    for market in &markets {
        tracing::info!(
            id = %market.id,
            question = %market.question,
            yes = market.yes_price(),
            liquidity = market.liquidity_num,
            "Market"
        );

        // TODO: Replace with LLM-based probability estimation
        let estimate = ProbabilityEstimate {
            market_id: market.id.clone(),
            estimated_probability: market.yes_price(), // placeholder: no edge
            confidence: 0.5,
            rationale: "placeholder estimate".to_string(),
        };

        if let Some(signal) = mispricing::evaluate(market, &estimate, bankroll, &config) {
            tracing::info!(
                market = %signal.market_id,
                side = ?signal.side,
                edge = signal.edge,
                size = signal.position_size_pct,
                "Trade signal"
            );
        }
    }

    Ok(())
}
