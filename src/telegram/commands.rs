//! Telegram command dispatch: maps bot command strings to reply strings.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::config::AppConfig;
use crate::live::ScanStats;
use crate::scanner::live::LiveScanner;
use crate::storage::postgres::PgPortfolio;
use crate::telegram::notifier::TelegramNotifier;

/// Dispatch a single Telegram command and return the reply string.
///
/// All arguments are borrowed — no long-lived state is mutated here.
#[allow(clippy::too_many_arguments)]
pub async fn handle_command(
    cmd: &str,
    chat_id: &str,
    full_text: &str,
    first_name: Option<&str>,
    portfolio: &PgPortfolio,
    notifier: &TelegramNotifier,
    scanner: &LiveScanner,
    http: &reqwest::Client,
    _cfg: &AppConfig,
    stats: &Arc<ScanStats>,
    start: &Instant,
) -> String {
    match cmd {
        "start" => {
            let name = first_name.unwrap_or("there");
            format!(
                "👋 Hi {name}! I'm the Polymarket Signal Bot.\n\n\
                 Commands:\n\
                 /stats — portfolio statistics\n\
                 /open — open positions\n\
                 /brier — model accuracy\n\
                 /health — bot health\n\
                 /traders — followed traders\n\
                 /leaderboard — top traders\n\
                 /help — show commands"
            )
        }
        "stats" => match portfolio.stats_summary().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(err = %e, "Failed to build stats");
                "⚠️ Failed to load stats".to_string()
            }
        },
        "open" => match portfolio.open_bets_summary().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(err = %e, "Failed to build open bets");
                "⚠️ Failed to load open bets".to_string()
            }
        },
        "brier" => match portfolio.brier_summary().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(err = %e, "Failed to build brier");
                "⚠️ Failed to load model accuracy".to_string()
            }
        },
        "health" => {
            let uptime = start.elapsed();
            let hours = uptime.as_secs() / 3600;
            let mins = (uptime.as_secs() % 3600) / 60;
            let scans = stats.scans_completed.load(Ordering::Relaxed);
            let mkts = stats.markets_scanned.load(Ordering::Relaxed);
            let sigs = stats.signals_found.load(Ordering::Relaxed);
            let news = stats.news_new.load(Ordering::Relaxed);
            let model_line = match scanner.model_age_secs().await {
                Some(age) => {
                    let h = (age / 3600.0) as u64;
                    let m = ((age % 3600.0) / 60.0) as u64;
                    format!("\n🧠 Model age: {h}h {m}m")
                }
                None => "\n🧠 Model: unavailable".to_string(),
            };
            format!(
                "🏥 *Bot Health*\n\n\
                 ⏱ Uptime: {hours}h {mins}m\n\
                 🔄 Scans completed: {scans}\n\
                 🔍 Markets scanned: {mkts}\n\
                 📡 Signals found: {sigs}\n\
                 📰 News processed: {news}{model_line}",
            )
        }
        "follow" => {
            if !notifier.is_owner(chat_id) {
                "🔒 Only the bot owner can follow traders.".to_string()
            } else {
                let arg = full_text.split_whitespace().nth(1).unwrap_or("");
                if arg.is_empty() {
                    "Usage: `/follow <wallet_address>`\n\nTip: use /leaderboard to browse top traders — wallet addresses are shown there for easy copy.".to_string()
                } else {
                    let wallet = arg.to_string();
                    let short = &wallet[..8.min(wallet.len())];
                    let strat_key = format!("copy:{short}");
                    if let Err(e) = portfolio
                        .ensure_key(
                            &strat_key,
                            crate::cycles::copy_trade::COPY_TRADER_STARTING_BANKROLL,
                        )
                        .await
                    {
                        tracing::warn!(err = %e, "Failed to init copy trader bankroll");
                    }
                    let username =
                        crate::scanner::copy_trader::fetch_trader_username(http, &wallet).await;
                    let display = username.as_deref().unwrap_or(short);
                    match portfolio
                        .add_followed_trader(
                            &wallet,
                            username.as_deref(),
                            "manual",
                            None,
                            None,
                            None,
                        )
                        .await
                    {
                        Ok(()) => format!(
                            "✅ Now following *{display}* (`{short}...`)\n💰 Bankroll: €{:.0}",
                            crate::cycles::copy_trade::COPY_TRADER_STARTING_BANKROLL
                        ),
                        Err(e) => format!("⚠️ Failed to follow: {e}"),
                    }
                }
            }
        }
        "unfollow" => {
            if !notifier.is_owner(chat_id) {
                "🔒 Only the bot owner can unfollow traders.".to_string()
            } else {
                let arg = full_text.split_whitespace().nth(1).unwrap_or("");
                if arg.is_empty() {
                    "Usage: `/unfollow <wallet_address>`".to_string()
                } else {
                    match portfolio.deactivate_trader(arg).await {
                        Ok(()) => {
                            let short = &arg[..8.min(arg.len())];
                            format!("✅ Unfollowed `{short}...`")
                        }
                        Err(e) => format!("⚠️ Failed to unfollow: {e}"),
                    }
                }
            }
        }
        "traders" => match portfolio.traders_summary().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(err = %e, "Failed to build traders summary");
                "⚠️ Failed to load traders".to_string()
            }
        },
        "leaderboard" => {
            let (day_res, month_res, all_res) = tokio::join!(
                crate::scanner::copy_trader::fetch_leaderboard(http, "DAY"),
                crate::scanner::copy_trader::fetch_leaderboard(http, "MONTH"),
                crate::scanner::copy_trader::fetch_leaderboard(http, "ALL"),
            );
            match (day_res, month_res, all_res) {
                (Ok(day), Ok(month), Ok(all)) => {
                    crate::scanner::copy_trader::format_multi_leaderboard(&[
                        ("Today", day.as_slice()),
                        ("This Month", month.as_slice()),
                        ("All Time", all.as_slice()),
                    ])
                }
                (day_res, month_res, all_res) => {
                    if let Err(e) = day_res.as_ref() {
                        tracing::warn!(err = %e, "Failed to fetch DAY leaderboard");
                    }
                    if let Err(e) = month_res.as_ref() {
                        tracing::warn!(err = %e, "Failed to fetch MONTH leaderboard");
                    }
                    if let Err(e) = all_res.as_ref() {
                        tracing::warn!(err = %e, "Failed to fetch ALL leaderboard");
                    }
                    "⚠️ Could not fetch leaderboard — try again shortly.".to_string()
                }
            }
        }
        "help" => "📖 *Commands*\n\n\
                 /stats — portfolio statistics\n\
                 /open — open positions\n\
                 /brier — model accuracy\n\
                 /health — bot health & uptime\n\
                 /traders — followed traders\n\
                 /leaderboard — top Polymarket traders\n\
                 /follow — follow a trader (owner)\n\
                 /unfollow — unfollow a trader (owner)\n\
                 /help — this message"
            .to_string(),
        _ => format!("❓ Unknown command: /{cmd}\nTry /help"),
    }
}

#[cfg(test)]
mod tests {
    // Integration tests that exercise real command logic require a live DB and
    // Telegram notifier, so they live in the end-to-end test suite.  This block
    // confirms the module compiles and exports `handle_command` correctly.
    use super::*;

    fn _assert_handle_command_is_async() {
        // If this compiles, handle_command is correctly declared async.
        fn _check<T: std::future::Future>(_: T) {}
        fn _dummy(
            p: &PgPortfolio,
            n: &TelegramNotifier,
            s: &LiveScanner,
            h: &reqwest::Client,
            c: &AppConfig,
            st: &Arc<ScanStats>,
            i: &Instant,
        ) {
            _check(handle_command(
                "help", "0", "/help", None, p, n, s, h, c, st, i,
            ));
        }
    }
}
