//! Shared application state for the copy-trading bot.
//!
//! Bundles the cross-cutting dependencies (DB, notifier, HTTP client, config,
//! trader monitor) plus the failure queue sender, so cycles/loops take a single
//! `&AppState` instead of a growing list of individual params. Built once in
//! `run_live`, wrapped in `Arc`, and cloned into each spawned loop.

use std::sync::Arc;

use tokio::sync::mpsc::UnboundedSender;

use crate::config::CopyTradingConfig;
use crate::scanner::copy_trader::CopyTraderMonitor;
use crate::storage::postgres::PgPortfolio;
use crate::telegram::notifier::TelegramNotifier;

pub struct AppState {
    pub portfolio: Arc<PgPortfolio>,
    pub notifier: Arc<TelegramNotifier>,
    pub monitor: Arc<CopyTraderMonitor>,
    pub cfg: Arc<CopyTradingConfig>,
    pub http: reqwest::Client,
    /// Producer end of the in-memory prune queue: permanently-failed chat_ids
    /// from broadcasts are pushed here for the prune worker to deactivate.
    pub fail_tx: UnboundedSender<Vec<String>>,
}
