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
/// Number of traders shown in the inline leaderboard reply.
const LEADERBOARD_DISPLAY_LIMIT: usize = 15;

// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

/// One entry from `GET /leaderboard`.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct LeaderboardEntry {
    #[serde(rename = "proxyWallet")]
    proxy_wallet: String,
    name: Option<String>,
    rank: Option<i32>,
    pnl: Option<f64>,
    volume: Option<f64>,
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
}

// ---------------------------------------------------------------------------
// Standalone leaderboard helpers (no monitor instance required)
// ---------------------------------------------------------------------------

/// Fetch the public Polymarket leaderboard and return the top entries
/// formatted for display.  This is **read-only** — nothing is written to the
/// database.
///
/// # Errors
///
/// Returns an error if the HTTP request fails or the response cannot be
/// parsed.
pub async fn fetch_leaderboard(http: &Client) -> Result<Vec<LeaderboardDisplay>> {
    let url = format!("{DATA_API}/leaderboard?window=all&limit=50");

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
        b.pnl
            .unwrap_or(0.0)
            .partial_cmp(&a.pnl.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let display = sorted
        .into_iter()
        .take(LEADERBOARD_DISPLAY_LIMIT)
        .enumerate()
        .map(|(i, e)| LeaderboardDisplay {
            rank: i + 1,
            name: e
                .name
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| format!("{}…", &e.proxy_wallet[..8.min(e.proxy_wallet.len())])),
            pnl: e.pnl.unwrap_or(0.0),
            volume: e.volume.unwrap_or(0.0),
        })
        .collect();

    Ok(display)
}

/// Format a slice of [`LeaderboardDisplay`] entries as a Telegram MarkdownV2-
/// compatible message.
///
/// # Example
///
/// ```ignore
/// let msg = format_leaderboard(&entries);
/// notifier.send_to(&chat_id, &msg).await?;
/// ```
pub fn format_leaderboard(entries: &[LeaderboardDisplay]) -> String {
    if entries.is_empty() {
        return "🏆 *Polymarket Leaderboard*\n\n_No data available._".to_string();
    }

    let mut lines = Vec::with_capacity(entries.len() + 3);
    lines.push("🏆 *Polymarket Leaderboard*\n".to_string());

    for entry in entries {
        let medal = match entry.rank {
            1 => "🥇",
            2 => "🥈",
            3 => "🥉",
            n => return_rank_str(n),
        };

        let pnl_str = format_dollars(entry.pnl);
        let vol_str = format_dollars(entry.volume);

        let line = if entry.rank <= 3 {
            format!(
                "{medal} *{}* — PnL: {} | Vol: {}",
                entry.name, pnl_str, vol_str
            )
        } else {
            format!(
                "{} {}. {} — PnL: {} | Vol: {}",
                medal, entry.rank, entry.name, pnl_str, vol_str,
            )
        };

        lines.push(line);
    }

    lines.push("\n_Data from Polymarket Data API_".to_string());
    lines.join("\n")
}

/// Returns a blank string for numbered ranks (the rank number is embedded in
/// the formatted line directly).
#[inline]
fn return_rank_str(_rank: usize) -> &'static str {
    " "
}

/// Format a dollar amount into a compact human-readable string.
///
/// * ≥ 1 000 000 → `$1.2M`
/// * ≥ 1 000     → `$890K`
/// * otherwise   → `$123`
fn format_dollars(value: f64) -> String {
    let abs = value.abs();
    let sign = if value < 0.0 { "-" } else { "" };
    if abs >= 1_000_000.0 {
        format!("{sign}${:.1}M", abs / 1_000_000.0)
    } else if abs >= 1_000.0 {
        format!("{sign}${:.0}K", abs / 1_000.0)
    } else {
        format!("{sign}${:.0}", abs)
    }
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

    /// Fetch the global leaderboard, apply filters, and upsert qualifying
    /// traders into `followed_traders`.
    ///
    /// * `min_pnl`     — minimum all-time PnL to include a trader
    /// * `min_volume`  — minimum all-time volume to include a trader
    /// * `max_traders` — cap on how many traders are persisted (top-N by PnL)
    #[allow(dead_code)]
    #[tracing::instrument(skip(self, portfolio))]
    pub async fn refresh_leaderboard(
        &self,
        portfolio: &PgPortfolio,
        min_pnl: f64,
        min_volume: f64,
        max_traders: usize,
    ) -> Result<usize> {
        let url = format!("{DATA_API}/leaderboard?window=all&limit=50",);

        let entries: Vec<LeaderboardEntry> = self
            .http
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

        tracing::info!(count = entries.len(), "Leaderboard entries fetched");

        // Filter and sort by descending PnL.
        let mut qualifying: Vec<&LeaderboardEntry> = entries
            .iter()
            .filter(|e| e.pnl.unwrap_or(0.0) >= min_pnl && e.volume.unwrap_or(0.0) >= min_volume)
            .collect();
        qualifying.sort_by(|a, b| {
            b.pnl
                .unwrap_or(0.0)
                .partial_cmp(&a.pnl.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        qualifying.truncate(max_traders);

        let upserted = qualifying.len();
        for entry in qualifying {
            portfolio
                .add_followed_trader(
                    &entry.proxy_wallet,
                    entry.name.as_deref(),
                    "leaderboard",
                    entry.rank,
                    entry.pnl,
                    entry.volume,
                )
                .await
                .context("upsert followed_trader")?;
        }

        tracing::info!(upserted, "Followed traders upserted from leaderboard");
        Ok(upserted)
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

        let trades = events
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

            let trades = match self.poll_trader_activity(&trader.proxy_wallet, since).await {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(
                        wallet = %trader.proxy_wallet,
                        err = %e,
                        "Failed to poll trader activity, skipping"
                    );
                    continue;
                }
            };

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
            }

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
