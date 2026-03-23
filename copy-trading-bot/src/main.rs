#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Thin re-exports of polymarket-common modules so that local code can use
// `crate::data`, `crate::storage`, etc. unchanged.
pub use polymarket_common::data;
pub use polymarket_common::format;
pub use polymarket_common::metrics;
pub use polymarket_common::model;
pub use polymarket_common::pricing;
pub use polymarket_common::signal;
pub use polymarket_common::storage;
// telegram is local so that the commands sub-module is accessible via crate::telegram
mod telegram;

mod config;
mod cycles;
mod live;
mod scanner;

use anyhow::Result;
use config::CopyTradingConfig;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls CryptoProvider");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("copy_trading_bot=info".parse()?),
        )
        .with_ansi(true)
        .with_target(true)
        .init();

    dotenvy::dotenv().ok();

    let cfg = CopyTradingConfig::load()?;
    live::run_live(Arc::new(cfg)).await
}
