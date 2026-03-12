#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod backtest;
mod backtest_runner;
mod bayesian;
mod calibration;
mod config;
mod cycles;
mod data;
mod format;
mod live;
mod metrics;
mod model;
mod pricing;
mod scanner;
mod storage;
mod strategy;
mod telegram;

use anyhow::Result;
use config::AppConfig;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls CryptoProvider");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("polymarket_bot=info".parse()?),
        )
        .with_ansi(true)
        .with_target(true)
        .init();

    dotenvy::dotenv().ok();

    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "backtest" => backtest_runner::run().await,
        "test" => {
            let mut cfg = AppConfig::load()?;
            cfg.scan_interval_mins = 2;
            cfg.news_scan_interval_mins = 2;
            live::run_live(Arc::new(cfg)).await
        }
        _ => {
            let cfg = AppConfig::load()?;
            live::run_live(Arc::new(cfg)).await
        }
    }
}
