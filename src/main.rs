#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod backtest;
mod bayesian;
mod calibration;
mod config;
mod data;
mod model;
mod pricing;
mod scanner;
mod storage;
mod strategy;
mod telegram;

use anyhow::Result;
use config::AppConfig;
use scanner::live::SignalSource;
use scanner::ws::{ActivityAlert, MarketWatcher};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use storage::portfolio::{BetSide, NewBet};
use storage::postgres::PgPortfolio;
use strategy::StrategyProfile;
use tokio::sync::{RwLock, mpsc};

/// Shared scan stats for heartbeat reporting.
struct ScanStats {
    scans_completed: AtomicU64,
    markets_scanned: AtomicU64,
    news_total: AtomicU64,
    news_new: AtomicU64,
    signals_found: AtomicU64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Install rustls crypto provider before any TLS usage.
    // Both ring and aws-lc-rs may be in the dep tree; pick one explicitly.
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls CryptoProvider");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("polymarket_bot=info".parse()?),
        )
        .with_ansi(true)
        .with_target(true)
        .init();

    dotenvy::dotenv().ok();

    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "backtest" => run_backtest().await,
        "test" => {
            let mut cfg = AppConfig::load()?;
            cfg.scan_interval_mins = 2;
            cfg.news_scan_interval_mins = 2;
            run_live(Arc::new(cfg)).await
        }
        _ => {
            let cfg = AppConfig::load()?;
            run_live(Arc::new(cfg)).await
        }
    }
}

async fn run_backtest() -> Result<()> {
    use backtest::engine::{BacktestConfig, run_backtest};
    use data::crawler::{Crawler, CrawlerConfig};

    tracing::info!("Polymarket backtest starting...");

    let crawler = Crawler::new(CrawlerConfig {
        market_limit: 2000,
        crypto_only: true,
        min_volume: 500.0,
        min_ticks: 20,
        min_duration_hours: 4.0,
        ..CrawlerConfig::default()
    });
    let dataset = crawler.build_dataset().await?;

    if dataset.is_empty() {
        tracing::warn!("No historical markets found");
        return Ok(());
    }

    let yes_count = dataset.iter().filter(|m| m.resolved_yes).count();
    tracing::info!(
        markets = dataset.len(),
        yes = yes_count,
        no = dataset.len() - yes_count,
        avg_ticks = dataset.iter().map(|m| m.price_history.len()).sum::<usize>() / dataset.len(),
        "Dataset ready"
    );

    let config = BacktestConfig {
        starting_cash: 300.0,
        edge_threshold: 0.03,
        position_size_pct: 0.05,
        slippage_pct: 0.01,
        fee_pct: 0.02,
        entry_point: 0.2,
        min_lookback: 10,
    };

    // Strategy 1: SMA Crossover
    let sma = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 10 {
            return (price, 0.1);
        }
        let prices: Vec<f64> = observed.iter().map(|t| t.p).collect();
        let short_w = (n / 3).max(3);
        let sma_short: f64 = prices[n - short_w..].iter().sum::<f64>() / short_w as f64;
        let sma_long: f64 = prices.iter().sum::<f64>() / n as f64;
        let momentum = sma_short - sma_long;
        let est = (price + momentum * 3.0).clamp(0.05, 0.95);
        let conf = (momentum.abs() * 8.0).clamp(0.2, 0.9);
        (est, conf)
    });
    log_result("SMA Crossover", &config, &sma);

    // Strategy 2: Mean Reversion
    let mr = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 5 {
            return (price, 0.1);
        }
        let prices: Vec<f64> = observed.iter().map(|t| t.p).collect();
        let avg: f64 = prices.iter().sum::<f64>() / n as f64;
        let std_dev: f64 =
            (prices.iter().map(|p| (p - avg).powi(2)).sum::<f64>() / n as f64).sqrt();
        let deviation = price - avg;
        if std_dev < 0.01 || deviation.abs() < std_dev {
            return (price, 0.1);
        }
        let est = (price - deviation * 0.5).clamp(0.05, 0.95);
        let conf = ((deviation.abs() / std_dev) * 0.2).clamp(0.2, 0.8);
        (est, conf)
    });
    log_result("Mean Reversion", &config, &mr);

    // Strategy 3: Trend Following
    let tf = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 10 {
            return (price, 0.1);
        }
        let prices: Vec<f64> = observed.iter().map(|t| t.p).collect();
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean: f64 = prices.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &p) in prices.iter().enumerate() {
            let x = i as f64 - x_mean;
            num += x * (p - y_mean);
            den += x * x;
        }
        let slope = if den > 0.0 { num / den } else { 0.0 };
        let recent = &prices[n.saturating_sub(5)..];
        let rmean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let vol: f64 =
            (recent.iter().map(|p| (p - rmean).powi(2)).sum::<f64>() / recent.len() as f64).sqrt();
        if vol > 0.12 || slope.abs() < 0.001 {
            return (price, 0.1);
        }
        let est = (price + slope * n as f64 * 0.5).clamp(0.05, 0.95);
        let conf = (slope.abs() * n as f64 * 3.0).clamp(0.2, 0.85);
        (est, conf)
    });
    log_result("Trend Following", &config, &tf);

    // Strategy 4: Contrarian at extremes
    let ct = run_backtest(&dataset, &config, |observed, price| {
        let n = observed.len();
        if n < 5 {
            return (price, 0.1);
        }
        let distance = (price - 0.5).abs();
        if distance < 0.20 {
            return (price, 0.1);
        }
        let recent: Vec<f64> = observed.iter().rev().take(5).map(|t| t.p).collect();
        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let accel = (price - recent_avg).abs();
        if accel < 0.02 {
            return (price, 0.1);
        }
        let est = price + (0.5 - price) * 0.25;
        let conf = (distance * 1.5 + accel * 3.0).clamp(0.3, 0.85);
        (est, conf)
    });
    log_result("Contrarian Extremes", &config, &ct);

    Ok(())
}

fn log_result(
    name: &str,
    config: &backtest::engine::BacktestConfig,
    result: &backtest::engine::BacktestResult,
) {
    let m = &result.metrics;
    tracing::info!(
        strategy = name,
        start = format_args!("€{:.0}", config.starting_cash),
        end = format_args!("€{:.0}", result.portfolio.total_equity()),
        roi = format_args!("{:.2}%", m.roi * 100.0),
        trades = m.total_trades,
        wins = m.winning_trades,
        losses = m.total_trades - m.winning_trades,
        win_rate = format_args!("{:.1}%", m.win_rate * 100.0),
        pnl = format_args!("€{:.2}", m.total_pnl),
        max_dd = format_args!("{:.2}%", m.max_drawdown * 100.0),
        sharpe = format_args!("{:.3}", m.sharpe_ratio),
        brier = format_args!("{:.4}", m.brier_score),
        "Backtest result"
    );
}

async fn run_live(cfg: Arc<AppConfig>) -> Result<()> {
    tracing::info!(
        news_interval_mins = cfg.news_scan_interval_mins,
        housekeeping_interval_mins = cfg.scan_interval_mins,
        "Polymarket Signal Bot starting (dual-loop)..."
    );

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

    let notifier = Arc::new(telegram::notifier::TelegramNotifier::new(
        &cfg.telegram_bot_token,
        &cfg.telegram_chat_id,
    ));
    let scanner = Arc::new(
        scanner::live::LiveScanner::new(&cfg, pool)
            .await
            .expect("failed to init scanner"),
    );

    let strategies = Arc::new(StrategyProfile::from_config(&cfg.strategies));
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

    let _ = notifier
        .send(&format!(
            "🤖 *Polymarket Signal Bot started*\n\n\
             💰 Bankroll: `€{total_bankroll:.2}` (started: `€{starting:.2}`)\n\
             💵 PnL: `€{total_pnl:+.2}` | ROI: `{total_roi:+.1}%`\n\
             📊 Open: {open_count} | Record: {wins}W/{losses}L\n\n\
             {strat_details}\n\n\
             ⚙️ *Config:*\n\
             ⏱ News: every {news_min}min | Housekeeping: every {hk_min}min\n\
             🎯 Max {max_sig} signals/day | Kelly: {kelly:.0}%\n\
             🔍 Min edge: {edge:.0}% | Min volume: ${vol:.0}\n\
             🧠 Pipeline: {pipeline}\n\
             🛑 Stop-loss: {sl:.0}% | Exit: {exit_days}d before expiry",
            open_count = open_bets.len(),
            wins = resolved.iter().filter(|b| b.won == Some(true)).count(),
            losses = resolved.iter().filter(|b| b.won == Some(false)).count(),
            strat_details = strat_lines.join("\n"),
            news_min = cfg.news_scan_interval_mins,
            hk_min = cfg.scan_interval_mins,
            max_sig = cfg.max_signals_per_day,
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
            if let Err(e) = housekeeping_cycle(
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
            tokio::time::sleep(Duration::from_secs(hk_interval * 60)).await;
        }
    });

    // Spawn news scanning loop (faster cycle)
    let ns_portfolio = Arc::clone(&portfolio);
    let ns_notifier = Arc::clone(&notifier);
    let ns_scanner = Arc::clone(&scanner);
    let ns_cfg = Arc::clone(&cfg);
    let ns_stats = Arc::clone(&stats);
    let ns_strategies = Arc::clone(&strategies);
    let news_scan = tokio::spawn(async move {
        let mut seen_headlines: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        loop {
            if let Err(e) = news_scan_cycle(
                &ns_portfolio,
                &ns_notifier,
                &ns_scanner,
                &ns_cfg,
                &ns_stats,
                &ns_strategies,
                &mut seen_headlines,
            )
            .await
            {
                tracing::error!(err = %e, "News scan cycle failed");
            }
            tokio::time::sleep(Duration::from_secs(ns_cfg.news_scan_interval_mins * 60)).await;
        }
    });

    // Spawn Telegram command polling loop
    let cmd_portfolio = Arc::clone(&portfolio);
    let cmd_notifier = Arc::clone(&notifier);
    let cmd_stats = Arc::clone(&stats);
    let cmd_start = std::time::Instant::now();
    let command_loop = tokio::spawn(async move {
        loop {
            let commands = cmd_notifier.poll_commands().await;
            for (chat_id, cmd, username, first_name) in &commands {
                // Track the user
                if let Err(e) = cmd_portfolio
                    .upsert_telegram_user(chat_id, username.as_deref(), first_name.as_deref())
                    .await
                {
                    tracing::warn!(err = %e, "Failed to upsert telegram user");
                }

                let reply = match cmd.as_str() {
                    "start" => {
                        let name = first_name.as_deref().unwrap_or("there");
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
                    "stats" => match cmd_portfolio.stats_summary().await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!(err = %e, "Failed to build stats");
                            "⚠️ Failed to load stats".to_string()
                        }
                    },
                    "open" => match cmd_portfolio.open_bets_summary().await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!(err = %e, "Failed to build open bets");
                            "⚠️ Failed to load open bets".to_string()
                        }
                    },
                    "brier" => match cmd_portfolio.brier_summary().await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!(err = %e, "Failed to build brier");
                            "⚠️ Failed to load model accuracy".to_string()
                        }
                    },
                    "health" => {
                        let uptime = cmd_start.elapsed();
                        let hours = uptime.as_secs() / 3600;
                        let mins = (uptime.as_secs() % 3600) / 60;
                        let scans = cmd_stats.scans_completed.load(Ordering::Relaxed);
                        let mkts = cmd_stats.markets_scanned.load(Ordering::Relaxed);
                        let sigs = cmd_stats.signals_found.load(Ordering::Relaxed);
                        let news = cmd_stats.news_new.load(Ordering::Relaxed);
                        format!(
                            "🏥 *Bot Health*\n\n\
                             ⏱ Uptime: {hours}h {mins}m\n\
                             🔄 Scans completed: {scans}\n\
                             🔍 Markets scanned: {mkts}\n\
                             📡 Signals found: {sigs}\n\
                             📰 News processed: {news}",
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
                };

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
            if let Err(e) = heartbeat_cycle(
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

    // --- WebSocket price watcher ---
    let (alert_tx, mut alert_rx) = mpsc::channel::<ActivityAlert>(100);

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

    // Spawn alert processing loop — runs XGBoost on markets triggered by WS
    let al_scanner = Arc::clone(&scanner);
    let al_portfolio = Arc::clone(&portfolio);
    let al_notifier = Arc::clone(&notifier);
    let al_strategies = Arc::clone(&strategies);
    let al_cfg = Arc::clone(&cfg);
    let al_token_map = Arc::clone(&token_map);
    let alert_loop = tokio::spawn(async move {
        // Throttle: don't re-assess same market within 15 minutes
        let mut last_assessed: HashMap<String, std::time::Instant> = HashMap::new();
        // Global WS cooldown: max 1 WS-triggered bet per 10 minutes
        let mut last_ws_bet = std::time::Instant::now() - Duration::from_secs(600);
        // Max WS bets per day
        let mut ws_bets_today: usize = 0;
        let mut ws_bets_date = chrono::Utc::now().format("%Y-%m-%d").to_string();
        const MAX_WS_BETS_PER_DAY: usize = 3;

        while let Some(alert) = alert_rx.recv().await {
            // Reset daily counter
            let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
            if today != ws_bets_date {
                ws_bets_today = 0;
                ws_bets_date = today;
            }
            if ws_bets_today >= MAX_WS_BETS_PER_DAY {
                continue;
            }

            let map = al_token_map.read().await;
            let market_id = match map.get(&alert.asset_id) {
                Some(id) => id.clone(),
                None => continue,
            };
            drop(map);

            // Per-market throttle: 15 minutes
            let now = std::time::Instant::now();
            if let Some(last) = last_assessed.get(&market_id)
                && now.duration_since(*last) < Duration::from_secs(900)
            {
                continue;
            }
            // Global cooldown: 10 minutes between any WS bets
            if now.duration_since(last_ws_bet) < Duration::from_secs(600) {
                continue;
            }
            last_assessed.insert(market_id.clone(), now);

            // Check not already bet on
            let skip_ids = al_portfolio.open_bet_market_ids().await.unwrap_or_default();
            if skip_ids.contains(&market_id) {
                continue;
            }

            tracing::info!(
                market_id = &market_id[..16.min(market_id.len())],
                price = format_args!("{:.1}%", alert.price * 100.0),
                delta = format_args!("{:+.1}%", (alert.price - alert.prev_price) * 100.0),
                "WS alert → assessing market"
            );

            match al_scanner.assess_alert(&market_id, alert.price).await {
                Ok(Some(signal)) => {
                    tracing::info!(
                        market = %signal.question,
                        side = %signal.side,
                        edge = format_args!("+{:.1}%", signal.edge * 100.0),
                        "WS-triggered signal found"
                    );

                    // Process through strategies — only first matching strategy bets
                    for strat in al_strategies.iter() {
                        let sent = al_portfolio
                            .strategy_signals_today(&strat.name)
                            .await
                            .unwrap_or(0);
                        if sent >= strat.max_signals_per_day {
                            continue;
                        }

                        let accepted = match strat.evaluate(&signal) {
                            Some(a) => a,
                            None => continue,
                        };

                        let strat_bankroll = al_portfolio
                            .strategy_bankroll(&strat.name)
                            .await
                            .unwrap_or(0.0);
                        let raw_bet = strat_bankroll * accepted.kelly_size;
                        if raw_bet < strat.min_bet {
                            continue;
                        }

                        let slipped_price =
                            (signal.current_price * (1.0 + al_cfg.slippage_pct)).min(0.99);
                        let shares = raw_bet / slipped_price;
                        let fee = raw_bet * al_cfg.fee_pct;
                        let total_cost = raw_bet + fee;
                        if total_cost > strat_bankroll {
                            continue;
                        }

                        let source_str = match signal.source {
                            SignalSource::XgBoost => "xgboost",
                            SignalSource::LlmConsensus => "llm_consensus",
                        };
                        let new_bet = NewBet {
                            market_id: signal.market_id.clone(),
                            question: signal.question.clone(),
                            side: signal.side.clone(),
                            entry_price: signal.current_price,
                            slipped_price,
                            shares,
                            cost: raw_bet,
                            fee,
                            estimated_prob: signal.estimated_prob,
                            confidence: signal.confidence,
                            edge: signal.edge,
                            kelly_size: accepted.kelly_size,
                            reasoning: signal.reasoning.clone(),
                            end_date: signal.end_date.clone(),
                            context: Some(signal.context.clone()),
                            strategy: strat.name.clone(),
                            source: source_str.to_string(),
                        };

                        // Log prediction for Brier score tracking
                        let source = match signal.source {
                            SignalSource::XgBoost => "xgboost",
                            SignalSource::LlmConsensus => "llm_consensus",
                        };
                        let _ = al_portfolio
                            .log_prediction(
                                &signal.market_id,
                                source,
                                signal.prior,
                                signal.estimated_prob,
                                signal.estimated_prob,
                                signal.confidence,
                                signal.edge,
                            )
                            .await;

                        match al_portfolio.place_bet(&new_bet).await {
                            Ok(_) => {
                                last_ws_bet = std::time::Instant::now();
                                ws_bets_today += 1;
                                let new_bankroll = al_portfolio
                                    .strategy_bankroll(&strat.name)
                                    .await
                                    .unwrap_or(0.0);
                                let side_emoji = match signal.side {
                                    BetSide::Yes => "🟢 YES",
                                    BetSide::No => "🔴 NO",
                                };
                                let msg = format!(
                                    "⚡ *WS-Triggered Bet* ({label} {strat_name})\n\
                                     {signal_msg}\n\n\
                                     💸 Bet: *{side}* `€{cost:.2}` ({shares:.1} shares @ `{price:.1}¢`)\n\
                                     💰 Strategy bankroll: `€{bankroll:.2}`",
                                    label = strat.label(),
                                    strat_name = strat.name,
                                    side = side_emoji,
                                    signal_msg = signal.to_telegram_message(),
                                    cost = raw_bet,
                                    shares = shares,
                                    price = slipped_price * 100.0,
                                    bankroll = new_bankroll,
                                );
                                broadcast(&al_notifier, &al_portfolio, &msg).await;
                                break;
                            }
                            Err(e) => {
                                tracing::error!(err = %e, "Failed to place WS-triggered bet");
                            }
                        }
                    }
                }
                Ok(None) => {
                    tracing::debug!(
                        market_id = &market_id[..16.min(market_id.len())],
                        "WS alert — no edge"
                    );
                }
                Err(e) => {
                    tracing::warn!(err = %e, "WS alert assessment failed");
                }
            }
        }
    });

    tokio::select! {
        r = housekeeping => {
            tracing::error!("Housekeeping loop exited: {:?}", r);
        }
        r = news_scan => {
            tracing::error!("News scan loop exited: {:?}", r);
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
    }

    Ok(())
}

/// Housekeeping: resolve bets, daily reports, reset counters.
async fn housekeeping_cycle(
    portfolio: &PgPortfolio,
    notifier: &telegram::notifier::TelegramNotifier,
    scanner: &scanner::live::LiveScanner,
    stop_loss_pct: f64,
    exit_days_before_expiry: i64,
) -> Result<()> {
    portfolio.reset_daily_if_needed().await?;

    // Check if any open bets have resolved
    let open_ids = portfolio.open_bet_market_ids().await?;

    for market_id in &open_ids {
        tokio::time::sleep(Duration::from_millis(200)).await;
        match scanner.check_market_resolution(market_id).await {
            Ok(Some(yes_won)) => {
                // Resolve LLM estimates for calibration tracking
                if let Err(e) = portfolio.resolve_estimates(market_id, yes_won).await {
                    tracing::warn!(err = %e, "Failed to resolve LLM estimates");
                }
                if let Some(r) = portfolio.resolve_bet(market_id, yes_won).await? {
                    let emoji = if r.won { "✅" } else { "❌" };
                    let result_label = if r.won { "WON" } else { "LOST" };
                    let roi = if r.cost > 0.0 {
                        r.pnl / r.cost * 100.0
                    } else {
                        0.0
                    };
                    let msg = format!(
                        "{emoji} *Bet {result_label}*\n\n\
                         📋 _{question}_\n\
                         🎲 Side: *{side}* @ `{price:.1}¢`\n\
                         📈 Edge: `+{edge:.1}%` | Confidence: `{conf:.0}%`\n\
                         💵 Stake: `€{cost:.2}` → PnL: `€{pnl:+.2}` ({roi:+.0}%)\n\n\
                         💰 Bankroll: `€{bankroll:.2}`\n\
                         📊 Record: {wins}W / {losses}L | Total PnL: `€{total_pnl:+.2}`",
                        question = r.question,
                        side = r.side,
                        price = r.entry_price * 100.0,
                        edge = r.edge * 100.0,
                        conf = r.confidence * 100.0,
                        cost = r.cost,
                        pnl = r.pnl,
                        bankroll = r.bankroll,
                        wins = r.total_wins,
                        losses = r.total_losses,
                        total_pnl = r.total_pnl,
                    );
                    broadcast(notifier, portfolio, &msg).await;
                    tracing::info!(
                        market = %market_id,
                        result = result_label,
                        pnl = format_args!("€{:+.2}", r.pnl),
                        bankroll = format_args!("€{:.2}", r.bankroll),
                        "Bet resolved"
                    );
                }
            }
            Ok(None) => {} // still open
            Err(e) => {
                tracing::warn!(market = %market_id, err = %e, "Resolution check failed");
            }
        }
    }

    // Send daily report if we haven't today
    let has_bets = !portfolio.all_bets().await?.is_empty();
    if portfolio.should_send_daily_report().await? && has_bets {
        portfolio.take_snapshot().await?;
        let report = portfolio.daily_summary().await?;
        broadcast(notifier, portfolio, &report).await;
        portfolio.mark_daily_report_sent().await?;
        tracing::info!("Daily report sent");
    }

    // Reload calibration curve with any newly resolved data
    if let Err(e) = scanner.reload_calibration().await {
        tracing::warn!(err = %e, "Failed to reload calibration curve");
    }

    // Check open positions for early exit and report unrealized P&L
    let open_bets = portfolio.open_bets().await?;
    if !open_bets.is_empty() {
        let mut lines = Vec::new();
        let mut total_unrealized = 0.0;
        let mut total_cost = 0.0;

        for bet in &open_bets {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let current = match scanner.fetch_current_price(&bet.market_id).await {
                Ok(Some(p)) => p,
                Ok(None) => continue,
                Err(e) => {
                    tracing::debug!(market = %bet.market_id, err = %e, "Price fetch failed");
                    continue;
                }
            };

            // For YES bets: value = shares * current_price
            // For NO bets: value = shares * (1 - current_price)
            let current_value = match bet.side {
                BetSide::Yes => bet.shares * current,
                BetSide::No => bet.shares * (1.0 - current),
            };
            let unrealized = current_value - bet.cost;

            // --- Early exit checks ---
            let loss_pct = if bet.cost > 0.0 {
                -unrealized / bet.cost
            } else {
                0.0
            };
            let days_left = bet
                .end_date
                .as_ref()
                .map(|d| {
                    chrono::DateTime::parse_from_rfc3339(d)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .or_else(|_| {
                            chrono::NaiveDateTime::parse_from_str(d, "%Y-%m-%dT%H:%M:%SZ")
                                .map(|n| n.and_utc())
                        })
                        .map(|dt| (dt - chrono::Utc::now()).num_days())
                        .unwrap_or(999)
                })
                .unwrap_or(999);

            let exit_reason = if loss_pct >= stop_loss_pct {
                Some(format!(
                    "stop-loss triggered ({:.0}% loss > {:.0}% limit)",
                    loss_pct * 100.0,
                    stop_loss_pct * 100.0
                ))
            } else if days_left <= exit_days_before_expiry && unrealized < 0.0 {
                Some(format!(
                    "expiry exit ({days_left}d left, underwater €{unrealized:.2})"
                ))
            } else {
                None
            };

            if let Some(reason) = exit_reason {
                match portfolio.early_exit(bet.id, current, &reason).await {
                    Ok(Some(r)) => {
                        let msg = format!(
                            "\u{1f6a8} *Early Exit*\n\n\
                             \u{1f4cb} _{question}_\n\
                             \u{1f3b2} Side: *{side}* @ `{entry:.1}\u{00a2}` \u{2192} `{now:.1}\u{00a2}`\n\
                             \u{1f4a1} Reason: {reason}\n\
                             \u{1f4b5} PnL: `\u{20ac}{pnl:+.2}`\n\n\
                             \u{1f4b0} Bankroll: `\u{20ac}{bankroll:.2}`\n\
                             \u{1f4ca} Record: {wins}W / {losses}L | Total PnL: `\u{20ac}{total_pnl:+.2}`",
                            question = r.question,
                            side = r.side,
                            entry = r.entry_price * 100.0,
                            now = current * 100.0,
                            pnl = r.pnl,
                            bankroll = r.bankroll,
                            wins = r.total_wins,
                            losses = r.total_losses,
                            total_pnl = r.total_pnl,
                        );
                        broadcast(notifier, portfolio, &msg).await;
                        tracing::info!(
                            market = %bet.question,
                            reason = %reason,
                            pnl = format_args!("€{:+.2}", r.pnl),
                            "Early exit executed"
                        );
                    }
                    Ok(None) => {}
                    Err(e) => tracing::error!(err = %e, "Early exit failed"),
                }
                continue; // Skip adding to open positions report
            }

            total_unrealized += unrealized;
            total_cost += bet.cost;

            let price_change = current - bet.entry_price;
            let arrow = if price_change > 0.01 {
                "📈"
            } else if price_change < -0.01 {
                "📉"
            } else {
                "➡️"
            };

            lines.push(format!(
                "{arrow} {side} _{q}_ `{entry:.0}¢→{now:.0}¢` `€{pnl:+.2}`",
                side = bet.side,
                q = truncate_str(&bet.question, 35),
                entry = bet.entry_price * 100.0,
                now = current * 100.0,
                pnl = unrealized,
            ));
        }

        let roi = if total_cost > 0.0 {
            total_unrealized / total_cost * 100.0
        } else {
            0.0
        };

        let mut msg = format!(
            "📋 *Open Positions* ({count})\n\n\
             💰 Unrealized: `€{pnl:+.2}` ({roi:+.1}%)\n",
            count = open_bets.len(),
            pnl = total_unrealized,
        );
        for line in &lines {
            msg.push_str(&format!("\n{line}"));
        }

        // Cache unrealized PnL for /stats command
        let _ = portfolio
            .upsert_f64_pub("unrealized_pnl", total_unrealized)
            .await;
        let _ = portfolio.upsert_f64_pub("open_exposure", total_cost).await;

        broadcast(notifier, portfolio, &msg).await;
    }

    tracing::info!(open_bets = open_bets.len(), "Housekeeping cycle complete");
    Ok(())
}

/// News scanning: fetch news, match to markets, assess with LLM, place bets per strategy.
async fn news_scan_cycle(
    portfolio: &PgPortfolio,
    notifier: &telegram::notifier::TelegramNotifier,
    scanner: &scanner::live::LiveScanner,
    cfg: &AppConfig,
    stats: &ScanStats,
    strategies: &[StrategyProfile],
    seen_headlines: &mut std::collections::HashSet<String>,
) -> Result<()> {
    portfolio.reset_daily_if_needed().await?;
    for s in strategies {
        portfolio.reset_strategy_daily_if_needed(&s.name).await?;
    }

    let bankroll = portfolio.bankroll().await?;
    tracing::info!(
        bankroll = format_args!("€{bankroll:.2}"),
        "Starting news scan..."
    );

    let mut skip_ids = portfolio.open_bet_market_ids().await?;
    // Skip markets rejected in the last 6h (avoid wasting LLM calls)
    let rejected_ids = portfolio.recently_rejected_market_ids(6).await?;
    // Skip markets we already bet on and resolved
    let resolved_ids = portfolio.resolved_bet_market_ids().await?;
    skip_ids.extend(rejected_ids);
    skip_ids.extend(resolved_ids);
    let past_bets = portfolio.learning_summary().await?;

    match scanner.scan(&skip_ids, &past_bets, seen_headlines).await {
        Ok(result) => {
            stats.scans_completed.fetch_add(1, Ordering::Relaxed);
            stats
                .markets_scanned
                .fetch_add(result.markets_scanned as u64, Ordering::Relaxed);
            stats
                .news_total
                .fetch_add(result.news_total as u64, Ordering::Relaxed);
            stats
                .news_new
                .fetch_add(result.news_new as u64, Ordering::Relaxed);
            stats
                .signals_found
                .fetch_add(result.signals.len() as u64, Ordering::Relaxed);

            let mut bets_placed: Vec<String> = Vec::new();
            let mut strategy_rejections: Vec<String> = Vec::new();
            // One bet per market — first accepting strategy wins
            let mut bet_market_ids: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            for signal in &result.signals {
                if bet_market_ids.contains(&signal.market_id) {
                    continue;
                }
                for strat in strategies {
                    let sent = portfolio.strategy_signals_today(&strat.name).await?;
                    let remaining = strat.max_signals_per_day.saturating_sub(sent);
                    if remaining == 0 {
                        tracing::debug!(
                            strategy = %strat.name,
                            market = %signal.question,
                            sent = sent,
                            max = strat.max_signals_per_day,
                            "Strategy daily limit reached, skipping"
                        );
                        continue;
                    }

                    let accepted = match strat.evaluate(signal) {
                        Some(a) => a,
                        None => {
                            let eff_edge = signal.edge * signal.confidence;
                            tracing::info!(
                                strategy = %strat.name,
                                market = %signal.question,
                                eff_edge = format_args!("+{:.1}%", eff_edge * 100.0),
                                conf = format_args!("{:.0}%", signal.confidence * 100.0),
                                min_edge = format_args!("{:.0}%", strat.min_effective_edge * 100.0),
                                min_conf = format_args!("{:.0}%", strat.min_confidence * 100.0),
                                "Strategy rejected signal (below thresholds)"
                            );
                            strategy_rejections.push(format!(
                                "{} {}: edge {:.1}%/conf {:.0}%",
                                strat.label(),
                                truncate_str(&signal.question, 40),
                                eff_edge * 100.0,
                                signal.confidence * 100.0,
                            ));
                            continue;
                        }
                    };

                    let strat_bankroll = portfolio.strategy_bankroll(&strat.name).await?;
                    let raw_bet = strat_bankroll * accepted.kelly_size;
                    if raw_bet < strat.min_bet {
                        tracing::debug!(
                            strategy = %strat.name,
                            kelly_bet = format_args!("€{raw_bet:.2}"),
                            min_bet = format_args!("€{:.2}", strat.min_bet),
                            "Kelly bet below minimum, skipping"
                        );
                        strategy_rejections.push(format!(
                            "{} {}: kelly €{:.2} < min €{:.2}",
                            strat.label(),
                            truncate_str(&signal.question, 40),
                            raw_bet,
                            strat.min_bet,
                        ));
                        continue;
                    }
                    let bet_amount = raw_bet;
                    let slipped_price = (signal.current_price * (1.0 + cfg.slippage_pct)).min(0.99);
                    let shares = bet_amount / slipped_price;
                    let fee = bet_amount * cfg.fee_pct;
                    let total_cost = bet_amount + fee;

                    if total_cost > strat_bankroll {
                        tracing::warn!(
                            strategy = %strat.name,
                            bankroll = format_args!("€{strat_bankroll:.2}"),
                            cost = format_args!("€{total_cost:.2}"),
                            "Insufficient strategy bankroll"
                        );
                        strategy_rejections.push(format!(
                            "{} {}: cost €{:.2} > bankroll €{:.2}",
                            strat.label(),
                            truncate_str(&signal.question, 40),
                            total_cost,
                            strat_bankroll,
                        ));
                        continue;
                    }

                    let new_bet = NewBet {
                        market_id: signal.market_id.clone(),
                        question: signal.question.clone(),
                        side: signal.side.clone(),
                        entry_price: signal.current_price,
                        slipped_price,
                        shares,
                        cost: bet_amount,
                        fee,
                        estimated_prob: signal.estimated_prob,
                        confidence: signal.confidence,
                        edge: signal.edge,
                        kelly_size: accepted.kelly_size,
                        reasoning: signal.reasoning.clone(),
                        end_date: signal.end_date.clone(),
                        context: Some(signal.context.clone()),
                        strategy: strat.name.clone(),
                        source: match signal.source {
                            SignalSource::XgBoost => "xgboost".to_string(),
                            SignalSource::LlmConsensus => "llm_consensus".to_string(),
                        },
                    };

                    // Log prediction for Brier score tracking
                    let source = match signal.source {
                        SignalSource::XgBoost => "xgboost",
                        SignalSource::LlmConsensus => "llm_consensus",
                    };
                    let _ = portfolio
                        .log_prediction(
                            &signal.market_id,
                            source,
                            signal.prior,
                            signal.estimated_prob,
                            signal.estimated_prob,
                            signal.confidence,
                            signal.edge,
                        )
                        .await;

                    match portfolio.place_bet(&new_bet).await {
                        Ok(_bet_id) => {
                            let new_strat_bankroll =
                                portfolio.strategy_bankroll(&strat.name).await?;
                            let open_count = portfolio.open_bets().await?.len();
                            let strat_signals =
                                portfolio.strategy_signals_today(&strat.name).await?;
                            tracing::info!(
                                strategy = %strat.name,
                                market = %signal.question,
                                side = %signal.side,
                                cost = format_args!("€{bet_amount:.2}"),
                                edge = format_args!("+{:.1}%", signal.edge * 100.0),
                                bankroll = format_args!("€{new_strat_bankroll:.2}"),
                                "Bet placed"
                            );

                            bets_placed.push(format!(
                                "{} {} €{:.2}",
                                strat.label(),
                                truncate_str(&signal.question, 40),
                                bet_amount,
                            ));

                            let news_section = if !signal.context.news_headlines.is_empty() {
                                let headlines: Vec<String> = signal
                                    .context
                                    .news_headlines
                                    .iter()
                                    .take(3)
                                    .map(|h| format!("  • _{}_", truncate_str(h, 80)))
                                    .collect();
                                format!("\n📰 *Triggered by:*\n{}\n", headlines.join("\n"))
                            } else {
                                String::new()
                            };

                            let side_emoji = match signal.side {
                                BetSide::Yes => "🟢 YES",
                                BetSide::No => "🔴 NO",
                            };
                            let msg = format!(
                                "{label} *{strat_name}*\n\
                                 {signal}\n\
                                 {news}\n\
                                 💸 *Bet: {side}*\n\
                                 💵 Stake: `€{cost:.2}` ({shares:.1} shares @ `{price:.1}¢`)\n\
                                 🏷 Fees: `€{fee:.2}` (slippage + trading)\n\
                                 💰 Strategy bankroll: `€{bankroll:.2}`\n\
                                 📊 Open bets: {open} | Strategy signals: {today}/{max}",
                                label = strat.label(),
                                strat_name = strat.name,
                                signal = signal.to_telegram_message(),
                                news = news_section,
                                side = side_emoji,
                                cost = bet_amount,
                                shares = shares,
                                price = slipped_price * 100.0,
                                fee = fee,
                                bankroll = new_strat_bankroll,
                                open = open_count,
                                today = strat_signals,
                                max = strat.max_signals_per_day,
                            );
                            broadcast(notifier, portfolio, &msg).await;
                            // One bet per market — stop trying other strategies
                            bet_market_ids.insert(signal.market_id.clone());
                            break;
                        }
                        Err(e) => {
                            tracing::error!(
                                strategy = %strat.name,
                                err = %e,
                                "Failed to place bet in DB"
                            );
                        }
                    }

                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }

            // Persist rejected signals to DB for analysis
            if let Err(e) = portfolio.save_rejected_signals(&result.rejections).await {
                tracing::warn!(err = %e, "Failed to save rejected signals");
            }

            // Send scan cycle summary to Telegram
            if !bets_placed.is_empty() {
                // Source breakdown — show all sources including zeros
                let sources: String = result
                    .source_counts
                    .iter()
                    .map(|(name, count)| format!("{name}(`{count}`)"))
                    .collect::<Vec<_>>()
                    .join(" · ");

                let mut summary = format!(
                    "📡 {sources}\n\
                     📊 `{eligible}` markets, `{matched}` matched → `{assessed}` assessed\n\
                     🎯 `{bets}` bets placed, `{candidates}` candidates, `{gate_rej}` gate rejected",
                    eligible = result.markets_scanned,
                    matched = result.news_matched,
                    assessed = result.llm_assessed,
                    bets = bets_placed.len(),
                    candidates = result.signals.len(),
                    gate_rej = result.rejections.len(),
                );

                if !bets_placed.is_empty() {
                    summary.push_str("\n\n✅ *Bets placed:*");
                    for b in &bets_placed {
                        summary.push_str(&format!("\n  {b}"));
                    }
                }

                // Rejection details only logged, not sent to Telegram

                broadcast(notifier, portfolio, &summary).await;
            }
        }
        Err(e) => {
            tracing::error!(err = %e, "News scan failed");
        }
    }

    let signals_today = portfolio.signals_sent_today().await?;
    let open_count = portfolio.open_bets().await?.len();
    tracing::info!(
        signals_today = signals_today,
        open_bets = open_count,
        "News scan cycle complete"
    );
    Ok(())
}

/// Periodic heartbeat — lets you know the bot is alive.
async fn heartbeat_cycle(
    portfolio: &PgPortfolio,
    notifier: &telegram::notifier::TelegramNotifier,
    cfg: &AppConfig,
    stats: &ScanStats,
    strategies: &[StrategyProfile],
) -> Result<()> {
    let bankroll = portfolio.bankroll().await?;
    let open_count = portfolio.open_bets().await?.len();
    let signals_today = portfolio.signals_sent_today().await?;

    // Per-strategy bankroll breakdown
    let mut strat_lines = String::new();
    for s in strategies {
        let sb = portfolio.strategy_bankroll(&s.name).await.unwrap_or(0.0);
        let sent = portfolio.strategy_signals_today(&s.name).await.unwrap_or(0);
        strat_lines.push_str(&format!(
            "\n  {} {}: `€{:.2}` ({}/{})",
            s.label(),
            s.name,
            sb,
            sent,
            s.max_signals_per_day,
        ));
    }

    // Read and reset counters
    let scans = stats.scans_completed.swap(0, Ordering::Relaxed);
    let markets = stats.markets_scanned.swap(0, Ordering::Relaxed);
    let news_total = stats.news_total.swap(0, Ordering::Relaxed);
    let news_new = stats.news_new.swap(0, Ordering::Relaxed);
    let signals = stats.signals_found.swap(0, Ordering::Relaxed);

    let msg = format!(
        "💓 *Heartbeat*\n\n\
         ⏱ {scans} scans in the last {interval}min\n\
         🔍 {markets} markets scanned\n\
         📰 {news_total} news items ({news_new} new)\n\
         🎯 {signals} signals found\n\n\
         💰 Total: `€{bankroll:.2}` | Open: {open} | Today: {today}/{max}\n\
         📊 Strategies:{strat_lines}",
        interval = cfg.heartbeat_interval_mins,
        open = open_count,
        today = signals_today,
        max = cfg.max_signals_per_day,
    );

    broadcast(notifier, portfolio, &msg).await;
    tracing::info!("Heartbeat sent");
    Ok(())
}

/// Broadcast a message to the owner and all subscribers.
async fn broadcast(
    notifier: &telegram::notifier::TelegramNotifier,
    portfolio: &PgPortfolio,
    message: &str,
) {
    let subs = portfolio.telegram_subscribers().await.unwrap_or_default();
    notifier.broadcast(&subs, message).await;
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}...")
    }
}
