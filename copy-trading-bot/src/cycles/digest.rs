//! Daily top-traders digest (advisory).
//!
//! Once per day, at a configured UTC hour, broadcasts the month's top traders
//! to all subscribers as *suggestions* to consider copying. This never follows
//! anyone automatically — the reader chooses who to `/follow`.

use anyhow::Result;
use chrono::{Timelike, Utc};
use reqwest::Client;

use crate::config::CopyTradingConfig;
use crate::scanner::copy_trader::{fetch_leaderboard, format_top_traders_digest};
use crate::storage::postgres::PgPortfolio;
use crate::telegram::notifier::TelegramNotifier;

const LAST_DIGEST_KEY: &str = "last_digest_date";

/// Pure decision: send only if the digest hasn't already gone out today AND the
/// clock has reached the configured hour.
pub fn should_send_digest(
    last: Option<&str>,
    today: &str,
    current_hour: u32,
    digest_hour: u32,
) -> bool {
    last != Some(today) && current_hour >= digest_hour
}

/// Send the daily digest if due. Best-effort: a fetch failure is logged and
/// retried on the next tick (the date marker is only written on success).
pub async fn maybe_send_daily_digest(
    http: &Client,
    portfolio: &PgPortfolio,
    notifier: &TelegramNotifier,
    cfg: &CopyTradingConfig,
) -> Result<()> {
    if !cfg.digest_enabled {
        return Ok(());
    }

    let now = Utc::now();
    let today = now.format("%Y-%m-%d").to_string();
    let last = portfolio.get_text_pub(LAST_DIGEST_KEY).await?;
    let last_opt = if last.is_empty() {
        None
    } else {
        Some(last.as_str())
    };

    if !should_send_digest(last_opt, &today, now.hour(), cfg.digest_hour_utc) {
        return Ok(());
    }

    match fetch_leaderboard(http, "MONTH").await {
        Ok(entries) => {
            let msg = format_top_traders_digest(&entries, cfg.digest_top_n);
            crate::live::broadcast(notifier, portfolio, &msg).await;
            portfolio.upsert_text_pub(LAST_DIGEST_KEY, &today).await?;
            tracing::info!(top_n = cfg.digest_top_n, "Sent daily top-traders digest");
        }
        Err(e) => {
            tracing::warn!(err = %e, "Digest leaderboard fetch failed; will retry next tick");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waits_for_hour() {
        // never sent, but before the configured hour
        assert!(!should_send_digest(None, "2026-08-23", 8, 9));
    }

    #[test]
    fn test_sends_when_hour_reached_and_not_sent_today() {
        assert!(should_send_digest(None, "2026-08-23", 9, 9));
        assert!(should_send_digest(Some("2026-08-22"), "2026-08-23", 10, 9));
    }

    #[test]
    fn test_skips_when_already_sent_today() {
        assert!(!should_send_digest(Some("2026-08-23"), "2026-08-23", 12, 9));
    }
}
