//! Postgres connection bootstrap shared by both bots.

use std::str::FromStr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use sqlx::PgPool;
use sqlx::postgres::PgConnectOptions;

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
