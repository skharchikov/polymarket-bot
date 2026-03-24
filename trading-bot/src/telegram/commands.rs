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
    _chat_id: &str,
    _full_text: &str,
    first_name: Option<&str>,
    portfolio: &PgPortfolio,
    _notifier: &TelegramNotifier,
    scanner: &LiveScanner,
    _http: &reqwest::Client,
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
                 /help — show commands"
            )
        }
        "stats" => match portfolio.stats_summary_ml_only().await {
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
        "help" => "📖 *Commands*\n\n\
                 /stats — portfolio statistics\n\
                 /open — open positions\n\
                 /brier — model accuracy\n\
                 /health — bot health & uptime\n\
                 /help — this message"
            .to_string(),
        _ => format!("❓ Unknown command: /{cmd}\nTry /help"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_handle_command_is_async() {
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
