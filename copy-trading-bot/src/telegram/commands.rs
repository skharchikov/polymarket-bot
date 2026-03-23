//! Telegram command dispatch for the copy-trading bot.

use crate::config::CopyTradingConfig;
use crate::cycles::copy_trade::COPY_TRADER_STARTING_BANKROLL;
use crate::scanner::copy_trader::{
    CopyTraderMonitor, fetch_leaderboard, fetch_trader_username, format_multi_leaderboard,
};
use crate::storage::postgres::PgPortfolio;
use crate::telegram::notifier::TelegramNotifier;

/// Dispatch a single Telegram command and return the reply string.
#[allow(clippy::too_many_arguments)]
pub async fn handle_command(
    cmd: &str,
    chat_id: &str,
    full_text: &str,
    first_name: Option<&str>,
    portfolio: &PgPortfolio,
    notifier: &TelegramNotifier,
    _monitor: &CopyTraderMonitor,
    http: &reqwest::Client,
    _cfg: &CopyTradingConfig,
) -> String {
    match cmd {
        "start" => {
            let name = first_name.unwrap_or("there");
            format!(
                "👋 Hi {name}! I'm the Polymarket Copy Trading Bot.\n\n\
                 Commands:\n\
                 /opencopy — open copy-trade positions\n\
                 /traders — followed traders\n\
                 /leaderboard — top traders\n\
                 /follow — follow a trader (owner)\n\
                 /unfollow — unfollow a trader (owner)\n\
                 /help — show commands"
            )
        }
        "opencopy" => match portfolio.open_copy_summary().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(err = %e, "Failed to build copy positions");
                "⚠️ Failed to load copy positions".to_string()
            }
        },
        "traders" => match portfolio.traders_summary().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(err = %e, "Failed to build traders summary");
                "⚠️ Failed to load traders".to_string()
            }
        },
        "leaderboard" => {
            let (day_res, month_res, all_res) = tokio::join!(
                fetch_leaderboard(http, "DAY"),
                fetch_leaderboard(http, "MONTH"),
                fetch_leaderboard(http, "ALL"),
            );
            match (day_res, month_res, all_res) {
                (Ok(day), Ok(month), Ok(all)) => format_multi_leaderboard(&[
                    ("Today", day.as_slice()),
                    ("This Month", month.as_slice()),
                    ("All Time", all.as_slice()),
                ]),
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
                            &format!("bankroll:{strat_key}"),
                            COPY_TRADER_STARTING_BANKROLL,
                        )
                        .await
                    {
                        tracing::warn!(err = %e, "Failed to init copy trader bankroll");
                    }
                    let username = fetch_trader_username(http, &wallet).await;
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
                            COPY_TRADER_STARTING_BANKROLL
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
        "help" => "📖 *Commands*\n\n\
                 /opencopy — open copy-trade positions\n\
                 /traders — followed traders\n\
                 /leaderboard — top Polymarket traders\n\
                 /follow — follow a trader (owner)\n\
                 /unfollow — unfollow a trader (owner)\n\
                 /help — this message"
            .to_string(),
        _ => format!("❓ Unknown command: /{cmd}\nTry /help"),
    }
}
