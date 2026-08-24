use anyhow::Result;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use sqlx::PgPool;

use tokio::sync::mpsc::{self, UnboundedReceiver};

use crate::config::CopyTradingConfig;
use crate::cycles;
use crate::metrics;
use crate::scanner::copy_trader::CopyTraderMonitor;
use crate::state::AppState;
use crate::storage::postgres::PgPortfolio;
use crate::telegram::notifier::{BotKind, TelegramNotifier};

/// Broadcast a message to the owner and all active subscribers.
///
/// Pure send: `notifier.broadcast` reports which chats permanently failed, and we
/// enqueue those to the prune worker via `state.fail_tx`. No DB access here —
/// subscriber-lifecycle is the worker's job ([`run_prune_worker`]).
pub async fn broadcast(state: &AppState, message: &str) {
    let subs = state
        .portfolio
        .telegram_subscribers(state.notifier.bot_kind())
        .await
        .unwrap_or_default();
    let resp = state.notifier.broadcast(&subs, message).await;
    if !resp.failed.is_empty() {
        // Non-blocking enqueue; if the worker is gone the ids are simply
        // re-detected on the next broadcast.
        let _ = state.fail_tx.send(resp.failed);
    }
}

/// Prune worker: drains permanently-failed chat_ids from the in-memory queue and
/// deactivates them. Runs as its own task, fully separate from broadcasting.
///
/// Coalesces: on each wake it drains every batch currently queued into a single
/// `HashSet`, deduping chat_ids across broadcasts and collapsing them to one DB
/// write. Without this, a dead chat_id re-enqueued by every failing broadcast
/// would trigger a redundant (idempotent) UPDATE per occurrence.
pub async fn run_prune_worker(
    portfolio: Arc<PgPortfolio>,
    notifier: Arc<TelegramNotifier>,
    mut rx: UnboundedReceiver<Vec<String>>,
) {
    while let Some(first) = rx.recv().await {
        let mut ids: HashSet<String> = first.into_iter().collect();
        // Drain anything else already queued so duplicates collapse into one call.
        while let Ok(more) = rx.try_recv() {
            ids.extend(more);
        }
        if ids.is_empty() {
            continue;
        }
        let batch: Vec<String> = ids.into_iter().collect();
        match portfolio
            .deactivate_telegram_users(notifier.bot_kind(), &batch)
            .await
        {
            Ok(_) => tracing::info!(count = batch.len(), "Deactivated unreachable subscribers"),
            Err(e) => {
                tracing::warn!(err = %e, count = batch.len(), "Failed to deactivate pruned subscribers")
            }
        }
    }
}

pub async fn run_live(cfg: Arc<CopyTradingConfig>) -> Result<()> {
    tracing::info!(
        interval_mins = cfg.copy_trade_interval_mins,
        "Copy Trading Bot starting..."
    );

    // Start Prometheus metrics server
    metrics::init(cfg.metrics_port);

    let pool = {
        let mut attempts = 0;
        loop {
            match PgPool::connect(&cfg.database_url).await {
                Ok(p) => break p,
                Err(e) => {
                    attempts += 1;
                    if attempts >= 10 {
                        return Err(e.into());
                    }
                    tracing::warn!(attempt = attempts, err = %e, "DB connect failed, retrying in 3s...");
                    tokio::time::sleep(Duration::from_secs(3)).await;
                }
            }
        }
    };
    let portfolio = Arc::new(PgPortfolio::new(pool.clone()).await?);
    portfolio.run_migrations().await?;
    tracing::info!("Database connected and migrations applied");

    let notifier = Arc::new(TelegramNotifier::new(
        &cfg.telegram_bot_token,
        &cfg.telegram_chat_id,
        BotKind::Copy,
    ));

    let monitor = Arc::new(CopyTraderMonitor::new(
        reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .expect("failed to build HTTP client"),
    ));

    let _ = notifier
        .send(&format!(
            "👥 *Copy Trading Bot* started\n\n\
             ⏱ Poll interval: every {}min",
            cfg.copy_trade_interval_mins,
        ))
        .await;

    // Shared state + in-memory prune queue.
    let (fail_tx, fail_rx) = mpsc::unbounded_channel::<Vec<String>>();
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .expect("failed to build HTTP client");
    let state = Arc::new(AppState {
        portfolio: Arc::clone(&portfolio),
        notifier: Arc::clone(&notifier),
        monitor: Arc::clone(&monitor),
        cfg: Arc::clone(&cfg),
        http,
        fail_tx,
    });

    // Prune worker — deactivates permanently-failed subscribers off the queue.
    tokio::spawn(run_prune_worker(
        Arc::clone(&portfolio),
        Arc::clone(&notifier),
        fail_rx,
    ));

    // Telegram command polling loop
    let cmd_state = Arc::clone(&state);
    let command_loop = tokio::spawn(async move {
        loop {
            let commands = cmd_state.notifier.poll_commands().await;
            for (chat_id, cmd, username, first_name, full_text) in &commands {
                match cmd_state
                    .portfolio
                    .upsert_telegram_user(
                        cmd_state.notifier.bot_kind(),
                        chat_id,
                        username.as_deref(),
                        first_name.as_deref(),
                    )
                    .await
                {
                    Ok(true) => {
                        let uname = username.as_deref().unwrap_or("—");
                        let fname = first_name.as_deref().unwrap_or("—");
                        let _ = cmd_state
                            .notifier
                            .send(&format!(
                                "🆕 *New user joined*\n\n\
                                 👤 {fname} (@{uname})\n\
                                 🆔 `{chat_id}`"
                            ))
                            .await;
                    }
                    Ok(false) => {}
                    Err(e) => tracing::warn!(err = %e, "Failed to upsert telegram user"),
                }

                tracing::info!(cmd = cmd.as_str(), chat_id, "Handling Telegram command");
                let reply = crate::telegram::commands::handle_command(
                    cmd,
                    chat_id,
                    full_text,
                    first_name.as_deref(),
                    &cmd_state,
                )
                .await;

                if let Err(e) = cmd_state.notifier.send_to(chat_id, &reply).await {
                    tracing::warn!(err = %e, chat_id = chat_id, "Failed to reply to command");
                }
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    });

    // Copy trade main loop
    let ct_state = Arc::clone(&state);
    let copy_trade_loop = tokio::spawn(async move {
        loop {
            if let Err(e) = cycles::copy_trade_cycle(&ct_state).await {
                tracing::error!(err = %e, "Copy trade cycle failed");
            }
            tokio::time::sleep(Duration::from_secs(
                ct_state.cfg.copy_trade_interval_mins * 60,
            ))
            .await;
        }
    });

    // Housekeeping loop — resolves copy bets + daily digest
    let hk_state = Arc::clone(&state);
    let housekeeping_loop = tokio::spawn(async move {
        loop {
            if let Err(e) = cycles::housekeeping_cycle(&hk_state).await {
                tracing::error!(err = %e, "Copy housekeeping cycle failed");
            }
            if let Err(e) = cycles::digest::maybe_send_daily_digest(&hk_state).await {
                tracing::warn!(err = %e, "Daily digest check failed");
            }
            tokio::time::sleep(Duration::from_secs(5 * 60)).await;
        }
    });

    tokio::select! {
        _ = shutdown_signal() => {
            tracing::info!("Shutdown signal received, stopping gracefully...");
        }
        r = command_loop => {
            tracing::error!("Command loop exited: {:?}", r);
        }
        r = copy_trade_loop => {
            tracing::error!("Copy trade loop exited: {:?}", r);
        }
        r = housekeeping_loop => {
            tracing::error!("Housekeeping loop exited: {:?}", r);
        }
    }

    tracing::info!("Sending shutdown notification...");
    let _ = notifier
        .send("🛑 Copy Trading Bot shutting down gracefully")
        .await;

    Ok(())
}

/// Wait for SIGINT (Ctrl-C) or SIGTERM (docker stop).
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to register SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        ctrl_c.await.ok();
    }
}
