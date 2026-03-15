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
    let open_count = portfolio.open_bets().await?.len();
    let signals_today = portfolio.signals_sent_today().await?;

    // Per-strategy bankroll breakdown
    let mut strat_lines = String::new();
    for s in strategies {
        let sb = portfolio.strategy_bankroll(&s.name).await.unwrap_or(0.0);
        let sent = portfolio.strategy_signals_today(&s.name).await.unwrap_or(0);
        strat_lines.push_str(&format!(
            "\n  {} {}: `€{:.2}` ({}/{})",
            s.label(),
            s.name,
            sb,
            sent,
            s.max_signals_per_day,
        ));
    }

    // Read and reset counters
    let scans = stats.scans_completed.swap(0, Ordering::Relaxed);
    let markets = stats.markets_scanned.swap(0, Ordering::Relaxed);
    let news_total = stats.news_total.swap(0, Ordering::Relaxed);
    let news_new = stats.news_new.swap(0, Ordering::Relaxed);
    let signals = stats.signals_found.swap(0, Ordering::Relaxed);

    let msg = format!(
        "💓 *Heartbeat*\n\n\
         ⏱ {scans} scans in the last {interval}min\n\
         🔍 {markets} markets scanned\n\
         📰 {news_total} news items ({news_new} new)\n\
         🎯 {signals} signals found\n\n\
         💰 Total: `€{bankroll:.2}` | Open: {open} | Today: {today}/{max}\n\
         📊 Strategies:{strat_lines}",
        interval = cfg.heartbeat_interval_mins,
        open = open_count,
        today = signals_today,
        max = strategies
            .iter()
            .map(|s| s.max_signals_per_day)
            .sum::<usize>(),
    );

    notify_owner(notifier, &msg).await;
    metrics::record_heartbeat();
    metrics::record_total_bankroll(bankroll);
    metrics::record_open_bets(open_count as u64);
    tracing::info!("Heartbeat sent");
    Ok(())
}
