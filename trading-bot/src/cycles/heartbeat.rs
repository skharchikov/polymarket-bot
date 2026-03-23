use anyhow::Result;
use std::sync::atomic::Ordering;

use crate::config::AppConfig;
use crate::live::{ScanStats, notify_owner};
use crate::metrics;
use crate::storage::postgres::PgPortfolio;
use crate::strategy::StrategyProfile;
use crate::telegram::notifier::TelegramNotifier;

pub async fn heartbeat_cycle(
    portfolio: &PgPortfolio,
    notifier: &TelegramNotifier,
    cfg: &AppConfig,
    stats: &ScanStats,
    strategies: &[StrategyProfile],
) -> Result<()> {
    let bankroll = portfolio.bankroll().await?;
    let signals_today = portfolio.signals_sent_today().await?;

    // Read and reset counters
    let scans = stats.scans_completed.swap(0, Ordering::Relaxed);
    let markets = stats.markets_scanned.swap(0, Ordering::Relaxed);
    let news_total = stats.news_total.swap(0, Ordering::Relaxed);
    let news_new = stats.news_new.swap(0, Ordering::Relaxed);
    let signals = stats.signals_found.swap(0, Ordering::Relaxed);

    let msg = format!(
        "💓 *Heartbeat* ({interval}min)\n\n\
         ⏱ {scans} scans | 🔍 {markets} markets\n\
         📰 {news_total} news ({news_new} new) | 🎯 {signals} signals\n\
         💰 `€{bankroll:.2}` | Today: {today}/{max}",
        interval = cfg.heartbeat_interval_mins,
        today = signals_today,
        max = strategies
            .iter()
            .map(|s| s.max_signals_per_day)
            .sum::<usize>(),
    );

    notify_owner(notifier, &msg).await;
    metrics::record_heartbeat();
    metrics::record_total_bankroll(bankroll);
    metrics::record_open_bets(portfolio.open_bets().await?.len() as u64);
    tracing::info!("Heartbeat sent");
    Ok(())
}
