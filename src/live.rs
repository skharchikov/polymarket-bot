use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use sqlx::PgPool;
use tokio::sync::{RwLock, mpsc};

use crate::config::AppConfig;
use crate::cycles;
use crate::metrics;
use crate::scanner::ws::{ActivityAlert, MarketWatcher};
use crate::storage::postgres::PgPortfolio;
use crate::strategy::StrategyProfile;
use crate::telegram::notifier::TelegramNotifier;

/// Curated victory GIFs sent on winning bets.
const VICTORY_GIFS: &[&str] = &[
    "https://media.giphy.com/media/GBtcj090cj3bBSwgJB/giphy.gif",
    "https://media.giphy.com/media/Ud0jIDEksXLhSwufo7/giphy.gif",
    "https://media.giphy.com/media/l4Ep5XhbkPJrgN6JG/giphy.gif",
    "https://media.giphy.com/media/ddHhhUBn25cuQ/giphy.gif",
    "https://media.giphy.com/media/iJgoGwkqb1mmH1mES3/giphy.gif",
    "https://media.giphy.com/media/GxIdtANXpn3qL1FG25/giphy.gif",
    "https://media.giphy.com/media/fUQ4rhUZJYiQsas6WD/giphy.gif",
    "https://media.giphy.com/media/1dMNqVx9Kb12EBjFrc/giphy.gif",
    "https://media.giphy.com/media/RPwrO4b46mOdy/giphy.gif",
    "https://media.giphy.com/media/yoJC2JaiEMoxIhQhY4/giphy.gif",
    "https://media.giphy.com/media/Vu5UbNpjpqfMq2UFg0/giphy.gif",
    "https://media.giphy.com/media/IbZbQr1BiYyOwxySzW/giphy.gif",
];

pub fn random_victory_gif() -> &'static str {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as usize;
    VICTORY_GIFS[seed % VICTORY_GIFS.len()]
}

/// Shared scan stats for heartbeat reporting.
pub struct ScanStats {
    pub scans_completed: AtomicU64,
    pub markets_scanned: AtomicU64,
    pub news_total: AtomicU64,
    pub news_new: AtomicU64,
    pub signals_found: AtomicU64,
}

/// Broadcast a message to the owner and all subscribers.
pub async fn broadcast(notifier: &TelegramNotifier, portfolio: &PgPortfolio, message: &str) {
    let subs = portfolio.telegram_subscribers().await.unwrap_or_default();
    notifier.broadcast(&subs, message).await;
}

/// Send a message only to the bot owner — operational noise subscribers don't need.
pub async fn notify_owner(notifier: &TelegramNotifier, message: &str) {
    if let Err(e) = notifier.send(message).await {
        tracing::warn!(err = %e, "Failed to send owner-only message");
    }
}

pub async fn run_live(cfg: Arc<AppConfig>) -> Result<()> {
    tracing::info!(
        bet_scan_interval_mins = cfg.bet_scan_interval_mins,
        housekeeping_interval_mins = cfg.scan_interval_mins,
        "Polymarket Signal Bot starting (dual-loop)..."
    );

    // Start Prometheus metrics server and tokio runtime collector
    metrics::init(cfg.metrics_port);
    let runtime_monitor = tokio_metrics::RuntimeMonitor::new(&tokio::runtime::Handle::current());
    metrics::spawn_tokio_collector(runtime_monitor);

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

    // Backfill URLs for any bets that were placed before the url column existed
    if let Err(e) = portfolio.backfill_urls().await {
        tracing::warn!(err = %e, "URL backfill failed (non-fatal)");
    }

    let notifier = Arc::new(TelegramNotifier::new(
        &cfg.telegram_bot_token,
        &cfg.telegram_chat_id,
    ));
    let scanner = Arc::new(
        crate::scanner::live::LiveScanner::new(&cfg, pool)
            .await
            .expect("failed to init scanner"),
    );

    let strategies = Arc::new(StrategyProfile::from_config(&cfg));
    portfolio
        .init_strategy_bankrolls(&strategies, cfg.strategy_bankroll)
        .await?;
    let strat_names: Vec<&str> = strategies.iter().map(|s| s.name.as_str()).collect();
    tracing::info!(strategies = ?strat_names, "Strategies loaded");

    let open_bets = portfolio.open_bets().await?;
    let resolved = portfolio.resolved_bets().await?;
    let starting = portfolio.starting_bankroll().await?;
    let num_strategies = strategies.len() as f64;
    let starting_per_strat = if num_strategies > 0.0 {
        starting / num_strategies
    } else {
        0.0
    };

    let mut strat_lines = Vec::new();
    let mut total_bankroll = 0.0_f64;
    for strat in strategies.iter() {
        let s_bankroll = portfolio.strategy_bankroll(&strat.name).await?;
        let s_open = open_bets
            .iter()
            .filter(|b| b.strategy == strat.name)
            .count();
        let s_resolved: Vec<_> = resolved
            .iter()
            .filter(|b| b.strategy == strat.name)
            .collect();
        let s_wins = s_resolved.iter().filter(|b| b.won == Some(true)).count();
        let s_losses = s_resolved.iter().filter(|b| b.won == Some(false)).count();
        let s_pnl: f64 = s_resolved.iter().filter_map(|b| b.pnl).sum();
        let s_roi = if starting_per_strat > 0.0 {
            (s_bankroll - starting_per_strat) / starting_per_strat * 100.0
        } else {
            0.0
        };
        total_bankroll += s_bankroll;
        strat_lines.push(format!(
            "{} *{}*: `€{:.2}` ({:+.1}%) | PnL `€{:+.2}` | {}W/{}L | {} open",
            strat.label(),
            strat.name,
            s_bankroll,
            s_roi,
            s_pnl,
            s_wins,
            s_losses,
            s_open,
        ));
    }

    let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();
    let total_roi = if starting > 0.0 {
        (total_bankroll - starting) / starting * 100.0
    } else {
        0.0
    };
    metrics::record_total_bankroll(total_bankroll);
    metrics::record_open_bets(open_bets.len() as u64);

    let version = env!("CARGO_PKG_VERSION");
    let build_tag = env!("BUILD_TAG");
    let changelog = env!("BUILD_CHANGELOG");
    let version_line = if !build_tag.is_empty() && format!("v{version}") != build_tag {
        format!("`v{version}` (since {build_tag})")
    } else {
        format!("`v{version}`")
    };

    let changelog_section = if changelog == "No changes since last release" {
        String::new()
    } else {
        format!("\n\n📝 *Changes:*\n{changelog}")
    };

    let _ = notifier
        .send(&format!(
            "🤖 *Polymarket Signal Bot* {version_line}\
             {changelog_section}\n\n\
             💰 Bankroll: `€{total_bankroll:.2}` (started: `€{starting:.2}`)\n\
             💵 PnL: `€{total_pnl:+.2}` | ROI: `{total_roi:+.1}%`\n\
             📊 Open: {open_count} | Record: {wins}W/{losses}L\n\n\
             {strat_details}\n\n\
             ⚙️ *Config:*\n\
             ⏱ Bet scan: every {bet_min}min | News: {news_status} | HK: every {hk_min}min\n\
             🎯 Max {max_sig} signals/day (per strategy) | Kelly: {kelly:.0}%\n\
             🔍 Min edge: {edge:.0}% | Min volume: ${vol:.0}\n\
             🧠 Pipeline: {pipeline}\n\
             🛑 Stop-loss: {sl:.0}% | Exit: {exit_days}d before expiry",
            open_count = open_bets.len(),
            wins = resolved.iter().filter(|b| b.won == Some(true)).count(),
            losses = resolved.iter().filter(|b| b.won == Some(false)).count(),
            strat_details = strat_lines.join("\n"),
            bet_min = cfg.bet_scan_interval_mins,
            news_status = if cfg.news_enabled {
                format!("every {}min", cfg.news_scan_interval_mins)
            } else {
                "disabled".to_string()
            },
            hk_min = cfg.scan_interval_mins,
            max_sig = strategies
                .iter()
                .map(|s| s.max_signals_per_day)
                .max()
                .unwrap_or(0),
            kelly = cfg.kelly_fraction * 100.0,
            edge = cfg.min_effective_edge * 100.0,
            vol = cfg.min_volume,
            pipeline = if cfg.model_sidecar_url.is_empty() {
                format!(
                    "XGBoost → Bayesian (LLM fallback: `{}` {}-agent)",
                    cfg.llm_model, cfg.consensus_agents
                )
            } else {
                format!(
                    "Ensemble sidecar → Bayesian (LLM fallback: `{}` {}-agent)",
                    cfg.llm_model, cfg.consensus_agents
                )
            },
            sl = cfg.stop_loss_pct * 100.0,
            exit_days = cfg.exit_days_before_expiry,
        ))
        .await;

    let stats = Arc::new(ScanStats {
        scans_completed: AtomicU64::new(0),
        markets_scanned: AtomicU64::new(0),
        news_total: AtomicU64::new(0),
        news_new: AtomicU64::new(0),
        signals_found: AtomicU64::new(0),
    });

    // Spawn housekeeping loop (resolution checks, daily reports)
    let hk_portfolio = Arc::clone(&portfolio);
    let hk_notifier = Arc::clone(&notifier);
    let hk_scanner = Arc::clone(&scanner);
    let hk_interval = cfg.scan_interval_mins;
    let hk_stop_loss = cfg.stop_loss_pct;
    let hk_exit_days = cfg.exit_days_before_expiry;
    let housekeeping = tokio::spawn(async move {
        loop {
            let hk_start = std::time::Instant::now();
            if let Err(e) = cycles::housekeeping_cycle(
                &hk_portfolio,
                &hk_notifier,
                &hk_scanner,
                hk_stop_loss,
                hk_exit_days,
            )
            .await
            {
                tracing::error!(err = %e, "Housekeeping cycle failed");
            }
            metrics::record_duration("bot_housekeeping_duration_seconds", hk_start.elapsed());
            tokio::time::sleep(Duration::from_secs(hk_interval * 60)).await;
        }
    });

    // Spawn bet scanning loop (market scoring + betting, always on)
    let bs_portfolio = Arc::clone(&portfolio);
    let bs_notifier = Arc::clone(&notifier);
    let bs_scanner = Arc::clone(&scanner);
    let bs_cfg = Arc::clone(&cfg);
    let bs_stats = Arc::clone(&stats);
    let bs_strategies = Arc::clone(&strategies);
    let bet_scan = tokio::spawn(async move {
        let mut seen_headlines: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        loop {
            let scan_start = std::time::Instant::now();
            if let Err(e) = cycles::bet_scan_cycle(
                &bs_portfolio,
                &bs_notifier,
                &bs_scanner,
                &bs_cfg,
                &bs_stats,
                &bs_strategies,
                &mut seen_headlines,
            )
            .await
            {
                tracing::error!(err = %e, "Bet scan cycle failed");
            }
            metrics::record_duration("bot_scan_duration_seconds", scan_start.elapsed());
            tokio::time::sleep(Duration::from_secs(bs_cfg.bet_scan_interval_mins * 60)).await;
        }
    });

    // Shared HTTP client for commands that need outbound requests (e.g. /leaderboard).
    let cmd_http = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .expect("failed to build command HTTP client");

    // Spawn Telegram command polling loop
    let cmd_portfolio = Arc::clone(&portfolio);
    let cmd_notifier = Arc::clone(&notifier);
    let cmd_stats = Arc::clone(&stats);
    let cmd_scanner = Arc::clone(&scanner);
    let cmd_cfg = Arc::clone(&cfg);
    let cmd_start = std::time::Instant::now();
    let command_loop = tokio::spawn(async move {
        loop {
            let commands = cmd_notifier.poll_commands().await;
            for (chat_id, cmd, username, first_name, full_text) in &commands {
                // Track the user
                if let Err(e) = cmd_portfolio
                    .upsert_telegram_user(chat_id, username.as_deref(), first_name.as_deref())
                    .await
                {
                    tracing::warn!(err = %e, "Failed to upsert telegram user");
                }

                tracing::info!(cmd = cmd.as_str(), chat_id, "Handling Telegram command");
                let reply = crate::telegram::commands::handle_command(
                    cmd,
                    chat_id,
                    full_text,
                    first_name.as_deref(),
                    &cmd_portfolio,
                    &cmd_notifier,
                    &cmd_scanner,
                    &cmd_http,
                    &cmd_cfg,
                    &cmd_stats,
                    &cmd_start,
                )
                .await;

                if let Err(e) = cmd_notifier.send_to(chat_id, &reply).await {
                    tracing::warn!(err = %e, chat_id = chat_id, "Failed to reply to command");
                }
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    });

    // Spawn heartbeat loop
    let hb_interval = cfg.heartbeat_interval_mins;
    let hb_portfolio = Arc::clone(&portfolio);
    let hb_notifier = Arc::clone(&notifier);
    let hb_cfg = Arc::clone(&cfg);
    let hb_stats = Arc::clone(&stats);
    let hb_strategies = Arc::clone(&strategies);
    let heartbeat = tokio::spawn(async move {
        if hb_interval == 0 {
            // Disabled — park forever
            std::future::pending::<()>().await;
            return;
        }
        loop {
            tokio::time::sleep(Duration::from_secs(hb_interval * 60)).await;
            if let Err(e) = cycles::heartbeat_cycle(
                &hb_portfolio,
                &hb_notifier,
                &hb_cfg,
                &hb_stats,
                &hb_strategies,
            )
            .await
            {
                tracing::error!(err = %e, "Heartbeat failed");
            }
        }
    });

    // --- Copy trading loop ---
    let ct_portfolio = Arc::clone(&portfolio);
    let ct_notifier = Arc::clone(&notifier);
    let ct_scanner = Arc::clone(&scanner);
    let ct_cfg = Arc::clone(&cfg);
    let copy_trade_loop = tokio::spawn(async move {
        if !ct_cfg.copy_trade_enabled {
            tracing::info!("Copy trading disabled");
            std::future::pending::<()>().await;
            return;
        }
        let monitor = crate::scanner::copy_trader::CopyTraderMonitor::new(
            reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("failed to build HTTP client"),
        );
        tracing::info!(
            interval_mins = ct_cfg.copy_trade_interval_mins,
            "Copy trading enabled"
        );
        loop {
            if let Err(e) = cycles::copy_trade_cycle(
                &ct_portfolio,
                &ct_notifier,
                &ct_scanner,
                &monitor,
                &ct_cfg,
            )
            .await
            {
                tracing::error!(err = %e, "Copy trade cycle failed");
            }
            tokio::time::sleep(Duration::from_secs(ct_cfg.copy_trade_interval_mins * 60)).await;
        }
    });

    // --- WebSocket price watcher ---
    let (alert_tx, alert_rx) = mpsc::channel::<ActivityAlert>(100);

    // Shared token→market_id mapping (refreshed periodically)
    let token_map: Arc<RwLock<HashMap<String, String>>> = Arc::new(RwLock::new(HashMap::new()));

    let ws_watcher = Arc::new(MarketWatcher::new(
        alert_tx, 0.03,  // 3% price move triggers alert
        500.0, // $500+ trade triggers alert
    ));

    // Spawn websocket connection
    let ws_watcher_run = Arc::clone(&ws_watcher);
    let ws_loop = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(15)).await;
        ws_watcher_run.run().await;
    });

    // Spawn token refresh loop — updates subscriptions every 30min
    let ws_tokens = Arc::clone(&ws_watcher.tokens);
    let ws_scanner = Arc::clone(&scanner);
    let ws_token_map = Arc::clone(&token_map);
    let ws_refresh = tokio::spawn(async move {
        loop {
            match ws_scanner.eligible_token_ids().await {
                Ok(pairs) => {
                    let mut map = ws_token_map.write().await;
                    map.clear();
                    let mut tokens = Vec::with_capacity(pairs.len());
                    for (token, market_id) in &pairs {
                        map.insert(token.clone(), market_id.clone());
                        tokens.push(token.clone());
                    }
                    drop(map);

                    *ws_tokens.write().await = tokens;
                    tracing::info!(count = pairs.len(), "Refreshed WS token subscriptions");
                }
                Err(e) => {
                    tracing::warn!(err = %e, "Failed to refresh WS tokens");
                }
            }
            tokio::time::sleep(Duration::from_secs(30 * 60)).await;
        }
    });

    // Spawn model age metric poller — records sidecar model age every 60s
    let ma_scanner = Arc::clone(&scanner);
    tokio::spawn(async move {
        loop {
            let age = ma_scanner.model_age_secs().await;
            crate::metrics::record_model_status(age);
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    });

    // Spawn model retrain monitor — checks sidecar after expected retrain window
    let mr_scanner = Arc::clone(&scanner);
    let mr_notifier = Arc::clone(&notifier);
    let retrain_interval = Duration::from_secs(cfg.retrain_interval_hours * 3600);
    let model_monitor = tokio::spawn(async move {
        // Get initial model age to know when next retrain is due
        tokio::time::sleep(Duration::from_secs(60)).await;
        let mut last_age = mr_scanner.model_age_secs().await.unwrap_or(0.0);

        loop {
            // Sleep until the model should have been retrained
            let time_until_retrain =
                retrain_interval.saturating_sub(Duration::from_secs_f64(last_age));
            // Check 20 min after expected retrain
            tokio::time::sleep(time_until_retrain + Duration::from_secs(20 * 60)).await;

            if let Some(age) = mr_scanner.model_age_secs().await {
                if age < last_age {
                    let h = (age / 3600.0) as u64;
                    let m = ((age % 3600.0) / 60.0) as u64;
                    let _ = mr_notifier
                        .send(&format!("🧠 *Model retrained* (age: {h}h {m}m)"))
                        .await;
                    tracing::info!(age_secs = age, "Model retrain detected");
                } else {
                    let _ = mr_notifier
                        .send("⚠️ *Model retrain overdue* — sidecar may have failed")
                        .await;
                    tracing::warn!(age_secs = age, "Model retrain overdue");
                }
                last_age = age;
            } else {
                let _ = mr_notifier.send("⚠️ *Model sidecar unreachable*").await;
                last_age = 0.0;
            }
        }
    });

    // Spawn alert processing loop — runs XGBoost on markets triggered by WS
    let alert_loop = tokio::spawn(cycles::alert_loop(
        alert_rx,
        Arc::clone(&scanner),
        Arc::clone(&portfolio),
        Arc::clone(&notifier),
        Arc::clone(&strategies),
        Arc::clone(&cfg),
        Arc::clone(&token_map),
    ));

    tokio::select! {
        _ = shutdown_signal() => {
            tracing::info!("Shutdown signal received, stopping gracefully...");
        }
        r = housekeeping => {
            tracing::error!("Housekeeping loop exited: {:?}", r);
        }
        r = bet_scan => {
            tracing::error!("Bet scan loop exited: {:?}", r);
        }
        r = heartbeat => {
            tracing::error!("Heartbeat loop exited: {:?}", r);
        }
        r = command_loop => {
            tracing::error!("Command loop exited: {:?}", r);
        }
        r = ws_loop => {
            tracing::error!("WebSocket loop exited: {:?}", r);
        }
        r = ws_refresh => {
            tracing::error!("WS refresh loop exited: {:?}", r);
        }
        r = alert_loop => {
            tracing::error!("Alert processing loop exited: {:?}", r);
        }
        r = model_monitor => {
            tracing::error!("Model monitor loop exited: {:?}", r);
        }
        r = copy_trade_loop => {
            tracing::error!("Copy trade loop exited: {:?}", r);
        }
    }

    tracing::info!("Sending shutdown notification...");
    let _ = notifier.send("🛑 Bot shutting down gracefully").await;

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
