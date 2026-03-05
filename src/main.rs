mod agent;
mod backtest;
mod data;
mod execution;
mod markets;
mod pricing;
mod risk;
mod scanner;
mod storage;
mod strategies;
mod telegram;

use anyhow::Result;
use chrono::{Datelike, Utc};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

const MAX_SIGNALS_PER_DAY: usize = 3;
const SCAN_INTERVAL_MINS: u64 = 30;

static SIGNALS_TODAY: AtomicUsize = AtomicUsize::new(0);
static LAST_RESET_DAY: AtomicUsize = AtomicUsize::new(0);

fn reset_daily_counter_if_needed() {
    let today = Utc::now().ordinal() as usize;
    let last = LAST_RESET_DAY.load(Ordering::Relaxed);
    if last != today {
        SIGNALS_TODAY.store(0, Ordering::Relaxed);
        LAST_RESET_DAY.store(today, Ordering::Relaxed);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("polymarket_bot=info".parse()?),
        )
        .init();

    dotenvy::dotenv().ok();

    tracing::info!("Polymarket Signal Bot starting...");

    let notifier = telegram::notifier::TelegramNotifier::from_env()?;
    let scanner = scanner::live::LiveScanner::new();

    // Send startup message
    let _ = notifier
        .send("🤖 *Polymarket Signal Bot started*\nScanning crypto markets for high-edge opportunities...")
        .await;

    loop {
        reset_daily_counter_if_needed();

        let remaining = MAX_SIGNALS_PER_DAY - SIGNALS_TODAY.load(Ordering::Relaxed);
        if remaining == 0 {
            tracing::info!("Daily signal limit reached, waiting for next day");
            tokio::time::sleep(Duration::from_secs(SCAN_INTERVAL_MINS * 60)).await;
            continue;
        }

        tracing::info!(remaining_signals = remaining, "Starting market scan...");

        match scanner.scan().await {
            Ok(signals) => {
                let to_send = signals.into_iter().take(remaining);

                for signal in to_send {
                    tracing::info!(
                        market = %signal.question,
                        edge = format_args!("+{:.1}%", signal.edge * 100.0),
                        score = format_args!("{:.4}", signal.score()),
                        "Sending YES signal"
                    );

                    let msg = signal.to_telegram_message();
                    match notifier.send(&msg).await {
                        Ok(()) => {
                            SIGNALS_TODAY.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            tracing::error!(err = %e, "Failed to send telegram signal");
                        }
                    }

                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
            Err(e) => {
                tracing::error!(err = %e, "Scan failed");
            }
        }

        tracing::info!(
            next_scan_mins = SCAN_INTERVAL_MINS,
            signals_sent_today = SIGNALS_TODAY.load(Ordering::Relaxed),
            "Scan cycle complete"
        );
        tokio::time::sleep(Duration::from_secs(SCAN_INTERVAL_MINS * 60)).await;
    }
}
