//! Copy-trader monitor: discovers top traders from the Polymarket leaderboard
//! and polls their on-chain activity to surface trade signals.
//!
//! This module is not yet wired into the main execution loop — it will be
//! integrated via the `copy_trade_cycle` in `src/cycles/copy_trade.rs`.

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use futures_util::future::join_all;
use reqwest::Client;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::time::{Duration, Instant};

use crate::storage::postgres::{NewCopyTradeEvent, PgPortfolio};

/// Trades older than this are skipped — price has likely moved too far.
const STALE_TRADE_SECS: i64 = 300; // 5 minutes

/// Polymarket Data API v2 (v1 was retired 2026-11-02 and now returns 410).
/// Docs: <https://docs.polymarket.com/api-reference/data-api/overview>
const DATA_API: &str = "https://data-api.polymarket.com/v2";
/// Default HTTP timeout for all data-API calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
/// Number of traders shown per period section in the inline leaderboard reply.
const LEADERBOARD_SECTION_LIMIT: usize = 5;
/// Max characters of a response body included in error messages.
const ERROR_BODY_SNIPPET: usize = 300;

// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

/// Every v2 response is wrapped as `{"data": ..., "pagination": {...}}`.
/// `data` is `null` (or `[]`) on a miss rather than a 404.
#[derive(Debug, Deserialize)]
struct DataEnvelope<T> {
    data: Option<T>,
}

/// Truncate `body` to [`ERROR_BODY_SNIPPET`] chars on a char boundary.
fn snippet(body: &str) -> String {
    let mut out: String = body.chars().take(ERROR_BODY_SNIPPET).collect();
    if out.len() < body.len() {
        out.push('…');
    }
    out
}

/// GET a Data API v2 URL and unwrap its `data` envelope.
///
/// Errors carry the endpoint, HTTP status, `Retry-After` (if any) and a body
/// snippet — the API's error body includes `error`, `code` and `trace_id`, so
/// a failure is diagnosable straight from the log line.
async fn get_data<T: DeserializeOwned>(http: &Client, what: &str, url: &str) -> Result<Option<T>> {
    let started = Instant::now();
    let resp = http
        .get(url)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .with_context(|| format!("{what}: GET {url} failed"))?;

    let status = resp.status();
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let body = resp
        .text()
        .await
        .with_context(|| format!("{what}: reading body of GET {url} failed (HTTP {status})"))?;
    let elapsed_ms = started.elapsed().as_millis() as u64;

    if !status.is_success() {
        let retry = retry_after
            .map(|r| format!(" retry-after={r}s"))
            .unwrap_or_default();
        bail!(
            "{what}: HTTP {status} from GET {url} after {elapsed_ms}ms{retry}: {}",
            snippet(&body)
        );
    }

    tracing::debug!(what, url, %status, elapsed_ms, bytes = body.len(), "Data API request ok");

    let env: DataEnvelope<T> = serde_json::from_str(&body)
        .with_context(|| format!("{what}: unexpected JSON from GET {url}: {}", snippet(&body)))?;
    Ok(env.data)
}

/// Map the legacy upper-case period names used by callers (`"DAY"`,
/// `"MONTH"`, `"ALL"`) to v2's lower-case `time_period` values.
fn v2_period(period: &str) -> String {
    period.to_ascii_lowercase()
}

// ---------------------------------------------------------------------------
// API response types (Data API v2: snake_case, numbers are JSON numbers)
// ---------------------------------------------------------------------------

/// One row from `GET /v2/leaderboard`.
#[derive(Debug, Deserialize)]
struct LeaderboardEntry {
    user_id: String,
    #[serde(default)]
    user_name: Option<String>,
    #[serde(default)]
    pnl: Option<f64>,
    #[serde(default)]
    volume: Option<f64>,
}

/// Single-wallet standing from `GET /v2/leaderboard?user=<wallet>`.
#[derive(Debug, Deserialize)]
struct LeaderboardUserEntry {
    #[serde(default)]
    user_name: Option<String>,
    #[serde(default)]
    pnl: Option<f64>,
    #[serde(default)]
    volume: Option<f64>,
    /// Rank by PnL; `0`/`null` means unranked.
    #[serde(default)]
    rank_pnl: Option<i64>,
}

/// One event from `GET /v2/activity`.
#[derive(Debug, Deserialize)]
struct ActivityEvent {
    /// Market slug — used to look up the Gamma numeric ID.
    slug: Option<String>,
    condition_id: Option<String>,
    /// "BUY" | "SELL" (empty for non-trade activity)
    side: Option<String>,
    /// Which market outcome token was traded: 0 = YES (first token), 1 = NO.
    outcome_index: Option<i64>,
    price: Option<f64>,
    /// Actual USD value of the trade (not shares).
    usdc_size: Option<f64>,
    #[serde(rename = "transaction_hash")]
    tx_hash: Option<String>,
    /// Epoch seconds.
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

/// Convert a per-wallet leaderboard response into [`TraderStats`].
/// Returns `None` when the wallet has no leaderboard entry (`data: null`).
fn parse_trader_stats(entry: Option<LeaderboardUserEntry>) -> Option<TraderStats> {
    let e = entry?;
    Some(TraderStats {
        rank: e
            .rank_pnl
            .filter(|r| *r > 0)
            .and_then(|r| i32::try_from(r).ok()),
        pnl: e.pnl.unwrap_or(0.0),
        volume: e.volume.unwrap_or(0.0),
        username: e.user_name.filter(|n| !n.is_empty()),
    })
}

/// Fetch stats (rank, PnL, volume, username) for a single wallet and time
/// period via `GET /v2/leaderboard?time_period=<period>&user=<wallet>`.
/// `period` is one of `"ALL"`, `"MONTH"`, `"DAY"`.
///
/// Returns `Ok(None)` when the wallet has no leaderboard entry.
pub async fn fetch_trader_stats(
    http: &Client,
    wallet: &str,
    period: &str,
) -> Result<Option<TraderStats>> {
    let url = format!(
        "{DATA_API}/leaderboard?time_period={}&user={wallet}",
        v2_period(period)
    );
    let entry: Option<LeaderboardUserEntry> = get_data(http, "trader stats", &url).await?;
    Ok(parse_trader_stats(entry))
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
            tracing::warn!(
                err = format!("{e:#}"),
                "Failed to load traders for stats refresh"
            );
            return;
        }
    };

    let fetches = traders.iter().map(|t| async move {
        // All-time drives rank/pnl/volume; MONTH supplies the last-month pnl.
        // Run both period fetches concurrently to halve per-trader latency.
        let (all, month) = tokio::join!(
            fetch_trader_stats(http, &t.proxy_wallet, "ALL"),
            fetch_trader_stats(http, &t.proxy_wallet, "MONTH"),
        );
        (t, all, month)
    });

    for (trader, res, month_res) in join_all(fetches).await {
        match res {
            Ok(Some(stats)) => {
                let pnl_month = match month_res {
                    Ok(m) => m.map(|m| m.pnl),
                    Err(e) => {
                        tracing::warn!(wallet = %trader.proxy_wallet, err = format!("{e:#}"), "Failed to fetch MONTH trader stats");
                        None
                    }
                };
                if let Err(e) = portfolio
                    .update_trader_stats(
                        &trader.proxy_wallet,
                        crate::storage::copy_trade::TraderStatsUpdate {
                            rank: stats.rank,
                            pnl: Some(stats.pnl),
                            pnl_month,
                            volume: Some(stats.volume),
                            username: stats.username.as_deref(),
                        },
                    )
                    .await
                {
                    tracing::warn!(wallet = %trader.proxy_wallet, err = format!("{e:#}"), "Failed to save trader stats");
                }
            }
            Ok(None) => {
                tracing::debug!(wallet = %trader.proxy_wallet, "No leaderboard entry for trader");
            }
            Err(e) => {
                tracing::warn!(wallet = %trader.proxy_wallet, err = format!("{e:#}"), "Failed to fetch trader stats");
            }
        }
    }
}

/// Fetch a trader's display name via the activity endpoint.
/// Returns `None` if the request fails or the trader has no activity.
pub async fn fetch_trader_username(http: &Client, wallet: &str) -> Option<String> {
    let url = format!("{DATA_API}/activity?user={wallet}&type=TRADE&limit=1");
    let rows: Vec<serde_json::Value> = match get_data(http, "trader username", &url).await {
        Ok(rows) => rows.unwrap_or_default(),
        Err(e) => {
            tracing::warn!(
                wallet,
                err = format!("{e:#}"),
                "Failed to fetch trader username"
            );
            return None;
        }
    };
    let name = rows.first()?["name"].as_str()?;
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
    let url = format!(
        "{DATA_API}/leaderboard?time_period={}&limit=10",
        v2_period(time_period)
    );
    let entries: Vec<LeaderboardEntry> = get_data(http, "leaderboard", &url)
        .await?
        .unwrap_or_default();

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
        .take(LEADERBOARD_SECTION_LIMIT)
        .enumerate()
        .map(|(i, e)| {
            let pnl = e.pnl.unwrap_or(0.0);
            let volume = e.volume.unwrap_or(0.0);
            let wallet = e.user_id;
            let name = e
                .user_name
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
        // v2 renamed `startTs` → `start` (epoch seconds, inclusive). Newest
        // first, default page of 100 — plenty for a 1-minute poll window.
        let url = format!("{DATA_API}/activity?user={wallet}&type=TRADE&start={since_ts}");

        let events: Vec<ActivityEvent> = get_data(&self.http, "trader activity", &url)
            .await?
            .unwrap_or_default();

        let raw_count = events.len();
        let trades: Vec<TraderTrade> = parse_activity_events(events);

        tracing::debug!(
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
            tracing::debug!(
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
                        err = format!("{e:#}"),
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
                    err = format!("{e:#}"),
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

    /// Deserialize a v2 `{"data": [...]}` activity body into events.
    fn activity(json: &str) -> Vec<ActivityEvent> {
        let env: DataEnvelope<Vec<ActivityEvent>> = serde_json::from_str(json).unwrap();
        env.data.unwrap_or_default()
    }

    // Real v2 response shape captured 2026-11-20 from
    // GET /v2/activity?user=<wallet>&type=TRADE&start=<ts>&limit=2
    const ACTIVITY_JSON: &str = r#"{"data":[
        {
            "proxy_wallet": "0x08458f7e9d2858027de579e4c3ca305475496b6f",
            "timestamp": 1795121488,
            "condition_id": "0x800376dc64e4e2376a495ed0df4a789192256c821bb4d1452acdd5931bca2fa5",
            "type": "TRADE",
            "size": 407.6,
            "usdc_size": 28.532,
            "transaction_hash": "0x8091362238",
            "price": 0.07,
            "token_id": "112320928806010548033202438997",
            "side": "BUY",
            "outcome_index": 0,
            "title": "Will Anthropic's market cap be less than $1.25T at market close on IPO day?",
            "slug": "will-anthropics-market-cap-be-less-than-1pt25t-at-market-close-on-ipo-day",
            "icon": "https://example.com/icon.png",
            "event_slug": "anthropic-ipo-closing-market-cap-higher-strikes",
            "outcome": "Yes",
            "name": "GollumGekko",
            "pseudonym": "Grumpy-Urn",
            "bio": "",
            "profile_image": "",
            "profile_image_optimized": ""
        },
        {
            "proxy_wallet": "0x08458f7e9d2858027de579e4c3ca305475496b6f",
            "timestamp": 1795121483,
            "condition_id": "0x65c3ff402d81e756af732fd67ea6521b15395206d2d77b8b2b006c212f620981",
            "type": "TRADE",
            "size": 88.857143,
            "usdc_size": 6.22,
            "transaction_hash": "0xbc7afc5f20",
            "price": 0.0699999999,
            "token_id": "87796361570300895",
            "side": "SELL",
            "outcome_index": 1,
            "title": "Other market",
            "slug": "other-market",
            "icon": "",
            "event_slug": "other-event",
            "outcome": "No",
            "name": "GollumGekko",
            "pseudonym": "Grumpy-Urn",
            "bio": "",
            "profile_image": "",
            "profile_image_optimized": ""
        }
    ],"pagination":{"limit":2,"offset":0,"has_more":true,"next_cursor":"abc"}}"#;

    #[test]
    fn test_parse_activity_events_real_shape() {
        let events = activity(ACTIVITY_JSON);
        assert_eq!(events.len(), 2, "should deserialize both events");

        let trades = parse_activity_events(events);
        assert_eq!(trades.len(), 2, "both trades should survive parsing");

        let t = &trades[0];
        assert_eq!(
            t.slug,
            "will-anthropics-market-cap-be-less-than-1pt25t-at-market-close-on-ipo-day"
        );
        assert_eq!(
            t.condition_id,
            "0x800376dc64e4e2376a495ed0df4a789192256c821bb4d1452acdd5931bca2fa5"
        );
        assert_eq!(t.side, "BUY");
        assert_eq!(t.outcome_index, 0);
        assert_eq!(t.price, 0.07);
        // usdc_size, not size (shares)
        assert_eq!(t.size_usd, 28.532);
        assert_eq!(t.tx_hash.as_deref(), Some("0x8091362238"));
        assert_eq!(t.timestamp.timestamp(), 1795121488);
        assert_eq!(trades[1].outcome_index, 1, "NO trade must be preserved");
    }

    /// `data: null` (a miss) must yield no events rather than an error.
    #[test]
    fn test_activity_null_data_is_empty() {
        let events = activity(r#"{"data":null}"#);
        assert!(events.is_empty());
    }

    /// The retired v1 camelCase shape must not silently parse as trades.
    #[test]
    fn test_v1_camel_case_fields_are_not_parsed() {
        let events = activity(
            r#"{"data":[{"slug":"m","conditionId":"0xabc","side":"BUY","price":0.5,"timestamp":1000}]}"#,
        );
        assert!(parse_activity_events(events).is_empty());
    }

    /// When the API omits outcome_index, default to 0 (YES).
    #[test]
    fn test_parse_activity_events_defaults_outcome_index() {
        let events = activity(
            r#"{"data":[{"slug":"some-market","condition_id":"0xabc","side":"BUY","price":0.5,"timestamp":1000}]}"#,
        );
        let trades = parse_activity_events(events);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].outcome_index, 0);
        assert_eq!(trades[0].size_usd, 0.0, "missing usdc_size defaults to 0");
    }

    #[test]
    fn test_parse_drops_events_missing_mandatory_fields() {
        // First event has no slug → dropped
        let events = activity(
            r#"{"data":[
                {"condition_id":"0xabc","side":"BUY","price":0.5,"usdc_size":10.0,"timestamp":1000},
                {"slug":"some-market","condition_id":"0xdef","side":"BUY","price":0.6,"usdc_size":20.0,"timestamp":2000}
            ]}"#,
        );
        let trades = parse_activity_events(events);
        assert_eq!(trades.len(), 1, "event with missing slug should be dropped");
        assert_eq!(trades[0].slug, "some-market");
    }

    #[test]
    fn test_parse_uses_usdc_size_not_shares() {
        let events = activity(
            r#"{"data":[{"slug":"market-a","condition_id":"0xabc","side":"SELL","price":0.9,"size":1000.0,"usdc_size":900.0,"timestamp":1000}]}"#,
        );
        let trades = parse_activity_events(events);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].size_usd, 900.0);
    }

    // Real v2 response shape captured 2026-11-20 from
    // GET /v2/leaderboard?time_period=month&user=<wallet>
    const TRADER_STATS_JSON: &str = r#"{"data":{
        "user_id": "0x08458f7e9d2858027de579e4c3ca305475496b6f",
        "pnl": -49909.12817958895,
        "volume": 1924516.784266,
        "rank_pnl": 683225,
        "rank_volume": 850,
        "user_name": "GollumGekko",
        "profile_image": "",
        "x_username": "GollumGekko1",
        "verified": false
    }}"#;

    fn user_entry(json: &str) -> Option<LeaderboardUserEntry> {
        serde_json::from_str::<DataEnvelope<LeaderboardUserEntry>>(json)
            .unwrap()
            .data
    }

    #[test]
    fn test_parse_trader_stats_real_shape() {
        let stats = parse_trader_stats(user_entry(TRADER_STATS_JSON)).expect("stats should parse");
        assert_eq!(stats.rank, Some(683_225));
        assert!((stats.pnl - -49_909.128_179_588_95).abs() < 1e-6);
        assert!((stats.volume - 1_924_516.784_266).abs() < 1e-6);
        assert_eq!(stats.username.as_deref(), Some("GollumGekko"));
    }

    #[test]
    fn test_parse_trader_stats_unknown_wallet() {
        // Wallet with no leaderboard entry → data: null → None
        assert!(parse_trader_stats(user_entry(r#"{"data":null}"#)).is_none());
    }

    #[test]
    fn test_parse_trader_stats_unranked_and_missing_fields() {
        // rank 0 = unranked; empty username → None
        let stats = parse_trader_stats(user_entry(
            r#"{"data":{"user_id":"0xabc","user_name":"","pnl":100.5,"volume":200.0,"rank_pnl":0}}"#,
        ))
        .expect("stats should parse");
        assert_eq!(stats.rank, None);
        assert_eq!(stats.pnl, 100.5);
        assert_eq!(stats.volume, 200.0);
        assert_eq!(stats.username, None);
    }

    #[test]
    fn test_leaderboard_board_shape() {
        let json = r#"{"data":[
            {"rank":1,"user_id":"0x2c33","pnl":3978314.97,"volume":226352898.6,"user_name":"whale","profile_image":"","x_username":"","verified":false},
            {"rank":2,"user_id":"0x4f1d","pnl":3297010.74,"volume":3797191.5,"user_name":"","verified":false}
        ],"pagination":{"limit":2,"offset":0,"has_more":true,"next_cursor":"x"}}"#;
        let rows: Vec<LeaderboardEntry> =
            serde_json::from_str::<DataEnvelope<Vec<LeaderboardEntry>>>(json)
                .unwrap()
                .data
                .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].user_id, "0x2c33");
        assert_eq!(rows[0].user_name.as_deref(), Some("whale"));
        assert_eq!(rows[1].pnl, Some(3297010.74));
    }

    #[test]
    fn test_v2_period_lowercases() {
        assert_eq!(v2_period("MONTH"), "month");
        assert_eq!(v2_period("ALL"), "all");
        assert_eq!(v2_period("DAY"), "day");
    }

    #[test]
    fn test_snippet_truncates_on_char_boundary() {
        let long = "é".repeat(ERROR_BODY_SNIPPET + 10);
        let out = snippet(&long);
        assert_eq!(out.chars().count(), ERROR_BODY_SNIPPET + 1);
        assert!(out.ends_with('…'));
        assert_eq!(snippet("short"), "short");
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
        let stats = fetch_trader_stats(&http, wallet, "ALL")
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
        // Active trader as of 2026-11-20
        let wallet = "0x08458f7e9d2858027de579e4c3ca305475496b6f";
        let since = chrono::Utc::now() - chrono::Duration::days(7);
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
