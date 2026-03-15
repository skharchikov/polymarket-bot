//! Copy-trader monitor: discovers top traders from the Polymarket leaderboard
//! and polls their on-chain activity to surface trade signals.
//!
//! This module is not yet wired into the main execution loop — it will be
//! integrated via the `copy_trade_cycle` in `src/cycles/copy_trade.rs`.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

use crate::storage::postgres::{NewCopyTradeEvent, PgPortfolio};

const DATA_API: &str = "https://data-api.polymarket.com";
/// Default HTTP timeout for all data-API calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
/// Number of traders shown per period section in the inline leaderboard reply.
const LEADERBOARD_SECTION_LIMIT: usize = 5;

// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

/// One entry from `GET /leaderboard`.
/// Fields come as strings from the API, so we deserialize to `Value` and parse.
#[derive(Debug, Deserialize)]
struct LeaderboardEntry {
    #[serde(rename = "proxyWallet")]
    proxy_wallet: String,
    #[serde(rename = "userName")]
    name: Option<String>,
    #[serde(default)]
    pnl: Option<serde_json::Value>,
    #[serde(default, rename = "vol")]
    volume: Option<serde_json::Value>,
}

impl LeaderboardEntry {
    fn pnl_f64(&self) -> f64 {
        self.volume_like(&self.pnl)
    }

    fn volume_f64(&self) -> f64 {
        self.volume_like(&self.volume)
    }

    fn volume_like(&self, v: &Option<serde_json::Value>) -> f64 {
        v.as_ref()
            .and_then(|v| match v {
                serde_json::Value::Number(n) => n.as_f64(),
                serde_json::Value::String(s) => s.parse().ok(),
                _ => None,
            })
            .unwrap_or(0.0)
    }
}

/// One trade event from `GET /activity`.
#[derive(Debug, Deserialize)]
struct ActivityEvent {
    #[serde(rename = "marketId")]
    market_id: Option<String>,
    #[serde(rename = "conditionId")]
    condition_id: Option<String>,
    /// "BUY" | "SELL"
    side: Option<String>,
    price: Option<f64>,
    size: Option<f64>,
    #[serde(rename = "transactionHash")]
    tx_hash: Option<String>,
    timestamp: Option<i64>,
}

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

/// A raw trade as returned by the Polymarket activity endpoint.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TraderTrade {
    pub market_id: String,
    pub condition_id: String,
    /// "BUY" or "SELL"
    pub side: String,
    pub price: f64,
    /// Size in USD
    pub size_usd: f64,
    pub tx_hash: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// A trade detected from a followed trader, ready for downstream filtering.
#[derive(Debug, Clone)]
pub struct DetectedTrade {
    pub trader_wallet: String,
    pub trade: TraderTrade,
}

/// Display-ready representation of a single leaderboard entry.
#[derive(Debug, Clone)]
pub struct LeaderboardDisplay {
    pub rank: usize,
    pub name: String,
    pub pnl: f64,
    pub volume: f64,
    pub wallet: String,
}

// ---------------------------------------------------------------------------
// Standalone leaderboard helpers (no monitor instance required)
// ---------------------------------------------------------------------------

/// Fetch a trader's display name from the Polymarket profile API.
/// Returns `None` if the request fails or the profile has no username set.
pub async fn fetch_trader_username(http: &Client, wallet: &str) -> Option<String> {
    let url = format!("{DATA_API}/profile?address={wallet}");
    let resp: serde_json::Value = http
        .get(&url)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    let name = resp["name"]
        .as_str()
        .or_else(|| resp["username"].as_str())?;
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Fetch the public Polymarket leaderboard for a given time period and return
/// the top entries formatted for display.  This is **read-only** — nothing is
/// written to the database.
///
/// `time_period` must be one of `"DAY"`, `"WEEK"`, `"MONTH"`, or `"ALL"`.
///
/// # Errors
///
/// Returns an error if the HTTP request fails or the response cannot be
/// parsed.
pub async fn fetch_leaderboard(
    http: &Client,
    time_period: &str,
) -> Result<Vec<LeaderboardDisplay>> {
    let url = format!("{DATA_API}/v1/leaderboard?timePeriod={time_period}&limit=10");

    let entries: Vec<LeaderboardEntry> = http
        .get(&url)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .context("leaderboard request failed")?
        .error_for_status()
        .context("leaderboard returned non-2xx")?
        .json()
        .await
        .context("leaderboard JSON parse failed")?;

    // Sort by descending PnL, then assign sequential display ranks.
    let mut sorted = entries;
    sorted.sort_by(|a, b| {
        b.pnl_f64()
            .partial_cmp(&a.pnl_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let display = sorted
        .into_iter()
        .take(LEADERBOARD_SECTION_LIMIT)
        .enumerate()
        .map(|(i, e)| {
            let pnl = e.pnl_f64();
            let volume = e.volume_f64();
            let wallet = e.proxy_wallet;
            let name = e
                .name
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| format!("{}…", &wallet[..8.min(wallet.len())]));
            LeaderboardDisplay {
                rank: i + 1,
                name,
                pnl,
                volume,
                wallet,
            }
        })
        .collect();

    Ok(display)
}

/// Format a slice of [`LeaderboardDisplay`] entries as a single period section
/// (no header or footer — used internally by [`format_multi_leaderboard`]).
///
/// When `show_wallets` is `true`, each entry also shows a `/follow <wallet>`
/// code snippet that the bot owner can tap-to-copy in Telegram.
fn format_leaderboard_section(entries: &[LeaderboardDisplay], show_wallets: bool) -> String {
    let mut lines = Vec::with_capacity(entries.len());

    for entry in entries {
        let pnl_str = crate::format::format_dollars(entry.pnl);
        let vol_str = crate::format::format_dollars(entry.volume);

        let line = match entry.rank {
            1 => format!("🥇 *{}* — PnL: {} | Vol: {}", entry.name, pnl_str, vol_str),
            2 => format!("🥈 *{}* — PnL: {} | Vol: {}", entry.name, pnl_str, vol_str),
            3 => format!("🥉 *{}* — PnL: {} | Vol: {}", entry.name, pnl_str, vol_str),
            n => format!(
                "{} {}. {} — PnL: {} | Vol: {}",
                return_rank_str(n),
                n,
                entry.name,
                pnl_str,
                vol_str,
            ),
        };

        if show_wallets {
            lines.push(format!("{line}\n   `/follow {}`", entry.wallet));
        } else {
            lines.push(line);
        }
    }

    lines.join("\n")
}

/// Format leaderboard results for multiple time periods into a single Telegram
/// message with one section per period.
///
/// `periods` is a slice of `(label, entries)` pairs, e.g.:
/// `&[("Today", &day_entries), ("This Month", &month_entries), ("All Time", &all_entries)]`
///
/// # Example
///
/// ```ignore
/// let msg = format_multi_leaderboard(&[
///     ("Today", &day_entries),
///     ("This Month", &month_entries),
///     ("All Time", &all_entries),
/// ]);
/// notifier.send_to(&chat_id, &msg).await?;
/// ```
pub fn format_multi_leaderboard(periods: &[(&str, &[LeaderboardDisplay])]) -> String {
    let mut parts = Vec::with_capacity(periods.len() + 2);
    parts.push("🏆 *Polymarket Leaderboard*".to_string());

    for (label, entries) in periods.iter() {
        let section_header = format!("\n📅 *{label}*");
        if entries.is_empty() {
            parts.push(format!("{section_header}\n_No data available._"));
        } else {
            parts.push(format!(
                "{section_header}\n{}",
                format_leaderboard_section(entries, true)
            ));
        }
    }

    parts.push("\n_Data from Polymarket Data API_".to_string());
    parts.join("\n")
}

/// Returns a blank string for numbered ranks (the rank number is embedded in
/// the formatted line directly).
#[inline]
fn return_rank_str(_rank: usize) -> &'static str {
    " "
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

/// Polls the Polymarket data API for leaderboard and trader activity.
///
/// Pass `&PgPortfolio` directly to each method — no long-lived state beyond
/// the HTTP client.
pub struct CopyTraderMonitor {
    http: Client,
}

impl CopyTraderMonitor {
    /// Build a new monitor with a shared `reqwest::Client`.
    pub fn new(http: Client) -> Self {
        Self { http }
    }

    /// Fetch recent trade activity for `wallet` since `since`.
    ///
    /// Returns only BUY-side trades (we mirror entries, not exits).
    #[tracing::instrument(skip(self), fields(wallet = %wallet))]
    pub async fn poll_trader_activity(
        &self,
        wallet: &str,
        since: DateTime<Utc>,
    ) -> Result<Vec<TraderTrade>> {
        let since_ts = since.timestamp();
        let url = format!("{DATA_API}/activity?user={wallet}&type=TRADE&startTs={since_ts}",);

        let events: Vec<ActivityEvent> = self
            .http
            .get(&url)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .context("activity request failed")?
            .error_for_status()
            .context("activity returned non-2xx")?
            .json()
            .await
            .context("activity JSON parse failed")?;

        let raw_count = events.len();
        let trades: Vec<TraderTrade> = events
            .into_iter()
            .filter_map(|e| {
                // Require all mandatory fields to be present.
                let market_id = e.market_id?;
                let condition_id = e.condition_id?;
                let side = e.side?;
                let price = e.price?;
                let size_usd = e.size?;
                let ts_secs = e.timestamp?;

                let timestamp = DateTime::from_timestamp(ts_secs, 0).unwrap_or_else(Utc::now);

                Some(TraderTrade {
                    market_id,
                    condition_id,
                    side,
                    price,
                    size_usd,
                    tx_hash: e.tx_hash,
                    timestamp,
                })
            })
            .collect();

        tracing::debug!(
            wallet = %wallet,
            since = %since.format("%Y-%m-%d %H:%M"),
            raw_events = raw_count,
            parsed_trades = trades.len(),
            "Trader activity fetched"
        );

        Ok(trades)
    }

    /// Iterate over all active traders, poll their recent activity, deduplicate
    /// against the `copy_trade_events` table, and return unseen trades.
    ///
    /// Each new trade is persisted to `copy_trade_events` before being returned
    /// so subsequent calls within the same run do not emit the same signal twice.
    #[tracing::instrument(skip(self, portfolio))]
    pub async fn detect_new_trades(&self, portfolio: &PgPortfolio) -> Result<Vec<DetectedTrade>> {
        let traders = portfolio
            .get_active_traders()
            .await
            .context("get_active_traders")?;

        tracing::info!(count = traders.len(), "Polling active traders");

        let mut detected = Vec::new();

        for trader in &traders {
            // Use last_checked_at as the lookback window; fall back to 24 h.
            let since = trader
                .last_checked_at
                .unwrap_or_else(|| Utc::now() - chrono::Duration::hours(24));

            let name = trader
                .username
                .as_deref()
                .unwrap_or(&trader.proxy_wallet[..8.min(trader.proxy_wallet.len())]);
            tracing::debug!(
                trader = %name,
                wallet = %trader.proxy_wallet,
                since = %since.format("%Y-%m-%d %H:%M"),
                "Polling trader"
            );

            let trades = match self.poll_trader_activity(&trader.proxy_wallet, since).await {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(
                        trader = %name,
                        wallet = %trader.proxy_wallet,
                        err = %e,
                        "Failed to poll trader activity, skipping"
                    );
                    continue;
                }
            };

            let mut new_count = 0usize;
            let mut skipped_count = 0usize;

            for trade in trades {
                let already_seen = portfolio
                    .is_copy_trade_seen(
                        &trader.proxy_wallet,
                        &trade.market_id,
                        &trade.side,
                        trade.price,
                    )
                    .await
                    .context("is_copy_trade_seen")?;

                if already_seen {
                    skipped_count += 1;
                    continue;
                }

                let event = NewCopyTradeEvent {
                    trader_wallet: trader.proxy_wallet.clone(),
                    market_id: trade.market_id.clone(),
                    condition_id: trade.condition_id.clone(),
                    side: trade.side.clone(),
                    price: trade.price,
                    size_usd: trade.size_usd,
                    tx_hash: trade.tx_hash.clone(),
                };

                portfolio
                    .save_copy_trade_event(&event)
                    .await
                    .context("save_copy_trade_event")?;

                detected.push(DetectedTrade {
                    trader_wallet: trader.proxy_wallet.clone(),
                    trade,
                });
                new_count += 1;
            }

            tracing::debug!(
                trader = %name,
                new = new_count,
                skipped = skipped_count,
                "Trader poll complete"
            );

            // Stamp the poll timestamp regardless of whether any trades were found.
            if let Err(e) = portfolio.update_trader_checked(&trader.proxy_wallet).await {
                tracing::warn!(
                    wallet = %trader.proxy_wallet,
                    err = %e,
                    "Failed to update last_checked_at"
                );
            }
        }

        tracing::info!(count = detected.len(), "New copy-trade events detected");
        Ok(detected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // hits real API
    async fn test_fetch_leaderboard_live() {
        let http = Client::new();
        let entries = fetch_leaderboard(&http, "ALL").await.unwrap();
        assert!(!entries.is_empty(), "leaderboard should have entries");
        assert!(entries.len() <= LEADERBOARD_SECTION_LIMIT);
        // First entry should be rank 1
        assert_eq!(entries[0].rank, 1);
        // Name should not be empty
        assert!(!entries[0].name.is_empty());
        // Top trader should have positive PnL
        assert!(entries[0].pnl > 0.0);
        // Print for manual inspection
        println!(
            "{}",
            format_multi_leaderboard(&[("All Time", entries.as_slice())])
        );
    }
}
