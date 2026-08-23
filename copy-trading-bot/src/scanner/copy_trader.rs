//! Copy-trader monitor: discovers top traders from the Polymarket leaderboard
//! and polls their on-chain activity to surface trade signals.
//!
//! This module is not yet wired into the main execution loop — it will be
//! integrated via the `copy_trade_cycle` in `src/cycles/copy_trade.rs`.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::future::join_all;
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

use crate::storage::postgres::{NewCopyTradeEvent, PgPortfolio};

/// Trades older than this are skipped — price has likely moved too far.
const STALE_TRADE_SECS: i64 = 300; // 5 minutes

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
    rank: Option<serde_json::Value>,
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

    /// Rank arrives as a string (e.g. `"80"`) — parse to i32, `None` if absent.
    fn rank_i32(&self) -> Option<i32> {
        self.rank.as_ref().and_then(|v| match v {
            serde_json::Value::Number(n) => n.as_i64().map(|n| n as i32),
            serde_json::Value::String(s) => s.parse().ok(),
            _ => None,
        })
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
    /// Market slug — used to look up the Gamma numeric ID.
    slug: Option<String>,
    #[serde(rename = "conditionId")]
    condition_id: Option<String>,
    /// "BUY" | "SELL"
    side: Option<String>,
    /// Which market outcome token was traded: 0 = YES (first token), 1 = NO.
    #[serde(rename = "outcomeIndex")]
    outcome_index: Option<i64>,
    price: Option<f64>,
    /// Actual USD value of the trade (not shares).
    #[serde(rename = "usdcSize")]
    usdc_size: Option<f64>,
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
    /// Market slug — used to look up the Gamma market (fetch_market_by_slug).
    pub slug: String,
    /// Hex condition ID — used for deduplication.
    pub condition_id: String,
    /// "BUY" or "SELL"
    pub side: String,
    /// Which market outcome the trade is on: 0 = YES token, 1 = NO token.
    /// Defaults to 0 when the API omits it.
    pub outcome_index: i64,
    pub price: f64,
    /// Size in USD (usdcSize from API).
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

/// Per-wallet stats from the Polymarket leaderboard endpoint.
#[derive(Debug, Clone)]
pub struct TraderStats {
    pub rank: Option<i32>,
    pub pnl: f64,
    pub volume: f64,
    pub username: Option<String>,
}

/// Extract [`TraderStats`] from a per-wallet leaderboard response.
/// Returns `None` when the wallet has no leaderboard entry.
fn parse_trader_stats(entries: Vec<LeaderboardEntry>) -> Option<TraderStats> {
    let e = entries.into_iter().next()?;
    Some(TraderStats {
        rank: e.rank_i32(),
        pnl: e.pnl_f64(),
        volume: e.volume_f64(),
        username: e.name.filter(|n| !n.is_empty()),
    })
}

/// Fetch all-time stats (rank, PnL, volume, username) for a single wallet via
/// `GET /v1/leaderboard?timePeriod=ALL&limit=1&user=<wallet>`.
///
/// Returns `Ok(None)` when the wallet has no leaderboard entry.
pub async fn fetch_trader_stats(http: &Client, wallet: &str) -> Result<Option<TraderStats>> {
    let url = format!("{DATA_API}/v1/leaderboard?timePeriod=ALL&limit=1&user={wallet}");
    let entries: Vec<LeaderboardEntry> = http
        .get(&url)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .context("trader stats request failed")?
        .error_for_status()
        .context("trader stats returned non-2xx")?
        .json()
        .await
        .context("trader stats JSON parse failed")?;
    Ok(parse_trader_stats(entries))
}

/// Refresh leaderboard stats for all active followed traders (best-effort).
///
/// Fetches per-wallet stats concurrently and writes them back to
/// `followed_traders`. Failures are logged and skipped so a partial refresh
/// never blocks the caller.
pub async fn refresh_followed_trader_stats(http: &Client, portfolio: &PgPortfolio) {
    let traders = match portfolio.get_active_traders().await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(err = %e, "Failed to load traders for stats refresh");
            return;
        }
    };

    let fetches = traders.iter().map(|t| async move {
        let res = fetch_trader_stats(http, &t.proxy_wallet).await;
        (t, res)
    });

    for (trader, res) in join_all(fetches).await {
        match res {
            Ok(Some(stats)) => {
                if let Err(e) = portfolio
                    .update_trader_stats(
                        &trader.proxy_wallet,
                        stats.rank,
                        Some(stats.pnl),
                        Some(stats.volume),
                        stats.username.as_deref(),
                    )
                    .await
                {
                    tracing::warn!(wallet = %trader.proxy_wallet, err = %e, "Failed to save trader stats");
                }
            }
            Ok(None) => {
                tracing::debug!(wallet = %trader.proxy_wallet, "No leaderboard entry for trader");
            }
            Err(e) => {
                tracing::warn!(wallet = %trader.proxy_wallet, err = %e, "Failed to fetch trader stats");
            }
        }
    }
}

/// Fetch a trader's display name via the activity endpoint.
/// Returns `None` if the request fails or the trader has no activity.
pub async fn fetch_trader_username(http: &Client, wallet: &str) -> Option<String> {
    let url = format!("{DATA_API}/activity?user={wallet}&type=TRADE&limit=1");
    let resp: serde_json::Value = http
        .get(&url)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    let name = resp.as_array()?.first()?["name"].as_str()?;
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
/// `time_period` must be one of `"DAY"`, `"MONTH"`, or `"ALL"`.
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
        let link = crate::format::profile_link(&entry.name, &entry.wallet);

        let line = match entry.rank {
            1 => format!("🥇 {link} — PnL: {pnl_str} | Vol: {vol_str}"),
            2 => format!("🥈 {link} — PnL: {pnl_str} | Vol: {vol_str}"),
            3 => format!("🥉 {link} — PnL: {pnl_str} | Vol: {vol_str}"),
            n => format!(
                "{} {}. {link} — PnL: {pnl_str} | Vol: {vol_str}",
                return_rank_str(n),
                n,
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

/// Format the top `n` leaderboard entries as a standalone "consider following"
/// digest message (MONTH window). Advisory only — the reader chooses who to
/// `/follow`; the bot never auto-follows.
pub fn format_top_traders_digest(entries: &[LeaderboardDisplay], n: usize) -> String {
    let top = &entries[..n.min(entries.len())];
    if top.is_empty() {
        return "📈 *Top Traders — This Month*\n\n_No leaderboard data available right now._"
            .to_string();
    }

    let mut parts = vec![
        "📈 *Top Traders — This Month*".to_string(),
        "Consider copying — tap a name for their profile:".to_string(),
        String::new(),
    ];
    for e in top {
        let link = crate::format::profile_link(&e.name, &e.wallet);
        let pnl = crate::format::format_dollars(e.pnl);
        let vol = crate::format::format_dollars(e.volume);
        parts.push(format!(
            "{}. {link} — PnL: {pnl} | Vol: {vol}\n   `/follow {}`",
            e.rank, e.wallet
        ));
    }
    parts.push(String::new());
    parts.push("_Advisory only — you choose who to /follow._".to_string());
    parts.join("\n")
}

// ---------------------------------------------------------------------------
// Activity parsing
// ---------------------------------------------------------------------------

/// Convert raw deserialized activity events into `TraderTrade`s, dropping any
/// entries that are missing mandatory fields.
fn parse_activity_events(events: Vec<ActivityEvent>) -> Vec<TraderTrade> {
    events
        .into_iter()
        .filter_map(|e| {
            let slug = e.slug?;
            let condition_id = e.condition_id?;
            let side = e.side?;
            let price = e.price?;
            let outcome_index = e.outcome_index.unwrap_or(0);
            let size_usd = e.usdc_size.unwrap_or(0.0);
            let ts_secs = e.timestamp?;
            let timestamp = DateTime::from_timestamp(ts_secs, 0).unwrap_or_else(Utc::now);
            Some(TraderTrade {
                slug,
                condition_id,
                side,
                outcome_index,
                price,
                size_usd,
                tx_hash: e.tx_hash,
                timestamp,
            })
        })
        .collect()
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
        let trades: Vec<TraderTrade> = parse_activity_events(events);

        tracing::info!(
            wallet = %wallet,
            since = %since.format("%Y-%m-%d %H:%M"),
            raw_events = raw_count,
            parsed_trades = trades.len(),
            "Trader activity fetched"
        );

        Ok(trades)
    }

    /// Iterate over all active traders, poll their recent activity in parallel,
    /// deduplicate against the `copy_trade_events` table, and return unseen trades.
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

        // Poll all traders concurrently.
        let poll_futures = traders.iter().map(|trader| {
            let since = trader
                .last_checked_at
                .unwrap_or_else(|| Utc::now() - chrono::Duration::hours(24));
            let name = trader
                .username
                .as_deref()
                .unwrap_or(&trader.proxy_wallet[..8.min(trader.proxy_wallet.len())])
                .to_string();
            tracing::info!(
                trader = %name,
                wallet = %trader.proxy_wallet,
                since = %since.format("%Y-%m-%d %H:%M"),
                "Polling trader"
            );
            async move {
                let result = self.poll_trader_activity(&trader.proxy_wallet, since).await;
                (trader, name, result)
            }
        });
        let poll_results = join_all(poll_futures).await;

        let now = Utc::now();
        let mut detected = Vec::new();

        // Process results sequentially for DB deduplication.
        for (trader, name, poll_result) in poll_results {
            let trades = match poll_result {
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
            let mut stale_count = 0usize;

            for trade in trades {
                // Skip trades that are too old — market price has likely moved.
                let age_secs = (now - trade.timestamp).num_seconds();
                if age_secs > STALE_TRADE_SECS {
                    stale_count += 1;
                    continue;
                }

                let already_seen = portfolio
                    .is_copy_trade_seen(
                        &trader.proxy_wallet,
                        &trade.condition_id,
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
                    market_id: trade.condition_id.clone(),
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

            tracing::info!(
                trader = %name,
                new = new_count,
                skipped = skipped_count,
                stale = stale_count,
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

    fn lb(rank: usize, name: &str, pnl: f64, wallet: &str) -> LeaderboardDisplay {
        LeaderboardDisplay {
            rank,
            name: name.to_string(),
            pnl,
            volume: pnl * 5.0,
            wallet: wallet.to_string(),
        }
    }

    #[test]
    fn test_format_top_traders_digest_basic() {
        let entries = vec![
            lb(1, "Alice", 120_000.0, "0xabc12345def"),
            lb(2, "Bob", 50_000.0, "0xdef67890abc"),
            lb(3, "Carol", 10_000.0, "0x11122233"),
        ];
        let out = format_top_traders_digest(&entries, 2);
        assert!(out.contains("Top Traders"));
        assert!(out.contains("Alice"));
        // profile link + follow hint present for the top entry
        assert!(out.contains("polymarket.com/profile/0xabc12345def"));
        assert!(out.contains("/follow 0xabc12345def"));
        // truncated to n = 2
        assert!(!out.contains("Carol"));
    }

    #[test]
    fn test_format_top_traders_digest_empty() {
        let out = format_top_traders_digest(&[], 5);
        assert!(out.contains("No leaderboard data"));
    }

    #[test]
    fn test_leaderboard_section_has_profile_link_and_follow() {
        let entries = vec![lb(1, "Alice", 120_000.0, "0xabc12345def")];
        let out = format_leaderboard_section(&entries, true);
        assert!(out.contains("polymarket.com/profile/0xabc12345def"));
        assert!(out.contains("/follow 0xabc12345def"));
    }

    // Real API response shape captured 2026-03-15
    const ACTIVITY_JSON: &str = r#"[
        {
            "proxyWallet": "0x37c1874a60d348903594a96703e0507c518fc53a",
            "timestamp": 1773601939,
            "conditionId": "0xfab8520004b4d201119f0362dc8678e8cf7f11b514efc48bc5a48aebf7974b50",
            "type": "TRADE",
            "size": 19.6,
            "usdcSize": 9.604,
            "transactionHash": "0x36b6c841eb1",
            "price": 0.49,
            "asset": "87207434043876055147",
            "side": "BUY",
            "outcomeIndex": 0,
            "title": "Spread: Trail Blazers (-8.5)",
            "slug": "nba-por-phi-2026-03-15-spread-away-8pt5",
            "icon": "https://example.com/icon.png",
            "eventSlug": "nba-por-phi-2026-03-15",
            "outcome": "Trail Blazers",
            "name": "CemeterySun",
            "pseudonym": "Pale-Bend",
            "bio": "",
            "profileImage": ""
        },
        {
            "proxyWallet": "0x37c1874a60d348903594a96703e0507c518fc53a",
            "timestamp": 1773601939,
            "conditionId": "0x65c3ff402d81e756af732fd67ea6521b15395206d2d77b8b2b006c212f620981",
            "type": "TRADE",
            "size": 1554.74,
            "usdcSize": 855.107,
            "transactionHash": "0x197d26499737",
            "price": 0.55,
            "asset": "87796361570300895",
            "side": "BUY",
            "outcomeIndex": 0,
            "title": "Spread: Bucks (-6.5)",
            "slug": "nba-ind-mil-2026-03-15-spread-home-6pt5",
            "icon": "https://example.com/icon2.png",
            "eventSlug": "nba-ind-mil-2026-03-15",
            "outcome": "Bucks",
            "name": "CemeterySun",
            "pseudonym": "Pale-Bend",
            "bio": "",
            "profileImage": ""
        }
    ]"#;

    /// Verify that the real API response shape deserializes correctly and all
    /// mandatory fields are extracted — this guards against the previous bug
    /// where `marketId` (non-existent) caused every trade to be dropped.
    #[test]
    fn test_parse_activity_events_real_shape() {
        let events: Vec<ActivityEvent> = serde_json::from_str(ACTIVITY_JSON).unwrap();
        assert_eq!(events.len(), 2, "should deserialize both events");

        let trades = parse_activity_events(events);
        assert_eq!(trades.len(), 2, "both trades should survive parsing");

        let t = &trades[0];
        assert_eq!(t.slug, "nba-por-phi-2026-03-15-spread-away-8pt5");
        assert_eq!(
            t.condition_id,
            "0xfab8520004b4d201119f0362dc8678e8cf7f11b514efc48bc5a48aebf7974b50"
        );
        assert_eq!(t.side, "BUY");
        assert_eq!(t.outcome_index, 0);
        assert_eq!(t.price, 0.49);
        // usdcSize, not size (shares)
        assert_eq!(t.size_usd, 9.604);
        assert_eq!(t.tx_hash.as_deref(), Some("0x36b6c841eb1"));
        assert_eq!(t.timestamp.timestamp(), 1773601939);
    }

    /// A trader buying the NO token (outcomeIndex 1) must be carried through so
    /// the copy-bot mirrors the correct side instead of always betting YES.
    #[test]
    fn test_parse_activity_events_carries_no_outcome() {
        let json = r#"[{
            "slug": "some-market",
            "conditionId": "0xabc",
            "side": "BUY",
            "price": 0.30,
            "usdcSize": 15.0,
            "outcomeIndex": 1,
            "timestamp": 1000
        }]"#;
        let events: Vec<ActivityEvent> = serde_json::from_str(json).unwrap();
        let trades = parse_activity_events(events);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].outcome_index, 1, "NO trade must be preserved");
    }

    /// When the API omits outcomeIndex, default to 0 (YES).
    #[test]
    fn test_parse_activity_events_defaults_outcome_index() {
        let json = r#"[{
            "slug": "some-market",
            "conditionId": "0xabc",
            "side": "BUY",
            "price": 0.5,
            "timestamp": 1000
        }]"#;
        let events: Vec<ActivityEvent> = serde_json::from_str(json).unwrap();
        let trades = parse_activity_events(events);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].outcome_index, 0);
    }

    #[test]
    fn test_parse_drops_events_missing_mandatory_fields() {
        // Missing slug → should be dropped
        let json = r#"[
            {"conditionId": "0xabc", "side": "BUY", "price": 0.5, "usdcSize": 10.0, "timestamp": 1000},
            {"slug": "some-market", "conditionId": "0xdef", "side": "BUY", "price": 0.6, "usdcSize": 20.0, "timestamp": 2000}
        ]"#;
        let events: Vec<ActivityEvent> = serde_json::from_str(json).unwrap();
        let trades = parse_activity_events(events);
        assert_eq!(trades.len(), 1, "event with missing slug should be dropped");
        assert_eq!(trades[0].slug, "some-market");
    }

    #[test]
    fn test_parse_uses_usdc_size_not_shares() {
        let json = r#"[{
            "slug": "market-a",
            "conditionId": "0xabc",
            "side": "SELL",
            "price": 0.9,
            "size": 1000.0,
            "usdcSize": 900.0,
            "timestamp": 1000
        }]"#;
        let events: Vec<ActivityEvent> = serde_json::from_str(json).unwrap();
        let trades = parse_activity_events(events);
        assert_eq!(trades.len(), 1);
        // Must be usdcSize (900), not size/shares (1000)
        assert_eq!(trades[0].size_usd, 900.0);
    }

    #[test]
    fn test_parse_usdc_size_defaults_to_zero_when_absent() {
        let json = r#"[{
            "slug": "market-b",
            "conditionId": "0xabc",
            "side": "BUY",
            "price": 0.5,
            "timestamp": 1000
        }]"#;
        let events: Vec<ActivityEvent> = serde_json::from_str(json).unwrap();
        let trades = parse_activity_events(events);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].size_usd, 0.0);
    }

    // Real API response shape captured 2026-07-15 from
    // GET /v1/leaderboard?timePeriod=ALL&limit=1&user=<wallet>
    const TRADER_STATS_JSON: &str = r#"[
        {
            "rank": "80",
            "proxyWallet": "0x37c1874a60d348903594a96703e0507c518fc53a",
            "userName": "CemeterySun",
            "xUsername": "",
            "verifiedBadge": false,
            "vol": 148559637.69049004,
            "pnl": 1927132.677116769,
            "profileImage": ""
        }
    ]"#;

    #[test]
    fn test_parse_trader_stats_real_shape() {
        let entries: Vec<LeaderboardEntry> = serde_json::from_str(TRADER_STATS_JSON).unwrap();
        let stats = parse_trader_stats(entries).expect("stats should parse");
        // rank arrives as a string — must be parsed to i32
        assert_eq!(stats.rank, Some(80));
        assert!((stats.pnl - 1_927_132.677_116_769).abs() < 1e-6);
        assert!((stats.volume - 148_559_637.690_490_04).abs() < 1e-6);
        assert_eq!(stats.username.as_deref(), Some("CemeterySun"));
    }

    #[test]
    fn test_parse_trader_stats_empty_response() {
        // Wallet with no leaderboard entry → empty array → None
        let entries: Vec<LeaderboardEntry> = serde_json::from_str("[]").unwrap();
        assert!(parse_trader_stats(entries).is_none());
    }

    #[test]
    fn test_parse_trader_stats_missing_optional_fields() {
        // rank absent, empty username → still returns pnl/vol, rank None, username None
        let json = r#"[{"proxyWallet": "0xabc", "userName": "", "pnl": 100.5, "vol": 200.0}]"#;
        let entries: Vec<LeaderboardEntry> = serde_json::from_str(json).unwrap();
        let stats = parse_trader_stats(entries).expect("stats should parse");
        assert_eq!(stats.rank, None);
        assert_eq!(stats.pnl, 100.5);
        assert_eq!(stats.volume, 200.0);
        assert_eq!(stats.username, None);
    }

    #[tokio::test]
    #[ignore] // hits real API
    async fn test_fetch_leaderboard_live() {
        let http = Client::new();
        let entries = fetch_leaderboard(&http, "ALL").await.unwrap();
        assert!(!entries.is_empty(), "leaderboard should have entries");
        assert!(entries.len() <= LEADERBOARD_SECTION_LIMIT);
        assert_eq!(entries[0].rank, 1);
        assert!(!entries[0].name.is_empty());
        assert!(entries[0].pnl > 0.0);
        println!(
            "{}",
            format_multi_leaderboard(&[("All Time", entries.as_slice())])
        );
    }

    #[tokio::test]
    #[ignore] // hits real API
    async fn test_fetch_trader_stats_live() {
        let http = Client::new();
        let wallet = "0x37c1874a60d348903594a96703e0507c518fc53a";
        let stats = fetch_trader_stats(&http, wallet)
            .await
            .unwrap()
            .expect("known trader should have stats");
        assert!(stats.pnl != 0.0, "pnl should be non-zero");
        assert!(stats.volume > 0.0, "volume should be positive");
        assert!(stats.rank.is_some(), "rank should be present");
        println!("{stats:?}");
    }

    #[tokio::test]
    #[ignore] // hits real API
    async fn test_poll_trader_activity_live() {
        let monitor = CopyTraderMonitor::new(Client::new());
        // Top leaderboard trader from 2026-03-15
        let wallet = "0x37c1874a60d348903594a96703e0507c518fc53a";
        let since = chrono::Utc::now() - chrono::Duration::hours(24);
        let trades = monitor.poll_trader_activity(wallet, since).await.unwrap();
        assert!(
            !trades.is_empty(),
            "active trader should have recent trades"
        );
        for t in &trades {
            assert!(!t.slug.is_empty(), "slug must be populated");
            assert!(!t.condition_id.is_empty(), "condition_id must be populated");
            assert!(t.price > 0.0 && t.price < 1.0, "price must be in (0,1)");
            assert!(t.size_usd >= 0.0, "size_usd must be non-negative");
        }
    }
}
