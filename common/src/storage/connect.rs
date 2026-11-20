//! Postgres connection bootstrap shared by both bots.

use std::str::FromStr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use sqlx::PgPool;
use sqlx::postgres::PgConnectOptions;

use crate::telegram::notifier::TelegramNotifier;

/// Number of connection attempts before giving up and letting the container
/// restart policy take over.
const MAX_ATTEMPTS: u32 = 10;
/// First retry delay; doubles each attempt up to [`MAX_BACKOFF`].
const INITIAL_BACKOFF: Duration = Duration::from_secs(2);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Connect to Postgres, retrying with exponential backoff.
///
/// Every attempt is logged with the target `host:port/database` (never the
/// password) and the full error chain, so DNS failures ("failed to lookup
/// address") are distinguishable from auth or refused-connection errors.
pub async fn connect_with_retry(database_url: &str) -> Result<PgPool> {
    let opts = PgConnectOptions::from_str(database_url)
        .context("DATABASE_URL is not a valid Postgres URL")?;
    let target = format!(
        "{}:{}/{}",
        opts.get_host(),
        opts.get_port(),
        opts.get_database().unwrap_or("<default>")
    );

    let started = Instant::now();
    let mut backoff = INITIAL_BACKOFF;
    for attempt in 1..=MAX_ATTEMPTS {
        match PgPool::connect_with(opts.clone()).await {
            Ok(pool) => {
                tracing::info!(
                    target_db = %target,
                    attempt,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "Connected to Postgres"
                );
                return Ok(pool);
            }
            Err(e) if attempt < MAX_ATTEMPTS => {
                tracing::warn!(
                    target_db = %target,
                    attempt,
                    max_attempts = MAX_ATTEMPTS,
                    retry_in_s = backoff.as_secs(),
                    err = format!("{:#}", anyhow::Error::from(e)),
                    "DB connect failed, retrying"
                );
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
            Err(e) => {
                return Err(anyhow::Error::from(e).context(format!(
                    "could not connect to Postgres at {target} after {MAX_ATTEMPTS} attempts ({}s)",
                    started.elapsed().as_secs()
                )));
            }
        }
    }
    unreachable!("loop always returns")
}

/// Connect to Postgres, or — when it stays unreachable after all retries —
/// alert the owner on Telegram and return `None`.
///
/// On `None` the caller should **halt in place** (stop all work and wait for
/// SIGTERM) rather than exit: the bots handle SIGTERM by exiting 0, so the
/// compose services must keep `restart: unless-stopped` to come back after a
/// host reboot, and exiting here would make Docker restart the bot and
/// re-send this alert every few minutes. Resume with
/// `docker compose restart <service>` once Postgres is back.
pub async fn connect_or_alert(
    database_url: &str,
    notifier: &TelegramNotifier,
    bot_name: &str,
) -> Option<PgPool> {
    let err = match connect_with_retry(database_url).await {
        Ok(pool) => return Some(pool),
        Err(e) => e,
    };

    let err_text = format!("{err:#}");
    tracing::error!(
        bot = bot_name,
        err = %err_text,
        "Database unavailable — bot halted. Fix Postgres, then `docker compose restart` this service"
    );

    let message = db_unavailable_message(bot_name, &err_text);
    for attempt in 1..=3u64 {
        match notifier.send(&message).await {
            Ok(()) => break,
            Err(e) => {
                tracing::warn!(
                    attempt,
                    err = format!("{e:#}"),
                    "Failed to send DB-down Telegram alert"
                );
                tokio::time::sleep(Duration::from_secs(5 * attempt)).await;
            }
        }
    }
    None
}

/// Telegram (Markdown v1) alert body. The error goes in a code block with
/// backticks stripped so `_`/`*` in it can't break the Markdown parse.
fn db_unavailable_message(bot_name: &str, err_text: &str) -> String {
    let err: String = err_text.replace('`', "'").chars().take(1500).collect();
    format!(
        "🚨 *{bot_name}: database unavailable — bot halted*\n\n\
         Gave up after {MAX_ATTEMPTS} connection attempts.\n\
         ```\n{err}\n```\n\
         Check: `docker compose ps postgres` and `docker compose logs --tail=50 postgres`\n\
         Then resume: `docker compose restart <service>`"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_unavailable_message_sanitises_backticks() {
        let msg = db_unavailable_message("Copy Trading Bot", "bad `thing` at postgres_host");
        assert!(msg.contains("Copy Trading Bot: database unavailable"));
        assert!(msg.contains("bad 'thing' at postgres_host"));
        // Only the fences/inline code we added contain backticks.
        assert_eq!(msg.matches('`').count(), 3 + 3 + 2 + 2 + 2);
    }
}
