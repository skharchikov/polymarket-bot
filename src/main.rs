#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod backtest;
mod calibration;
mod config;
mod data;
mod pricing;
mod scanner;
mod storage;
mod strategy;
mod telegram;

use anyhow::Result;
use config::AppConfig;
use sqlx::PgPool;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use storage::portfolio::NewBet;
use storage::postgres::PgPortfolio;
use strategy::StrategyProfile;

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

    let bankroll = portfolio.bankroll().await?;
    let open_count = portfolio.open_bets().await?.len();

    let starting = portfolio.starting_bankroll().await?;
    let resolved = portfolio.resolved_bets().await?;
    let wins = resolved.iter().filter(|b| b.won == Some(true)).count();
    let losses = resolved.iter().filter(|b| b.won == Some(false)).count();
    let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();

    let _ = notifier
        .send(&format!(
            "🤖 *Polymarket Signal Bot started*\n\n\
             💰 Bankroll: `€{bankroll:.2}` (started: `€{starting:.2}`)\n\
             📊 Open bets: {open_count} | Record: {wins}W/{losses}L\n\
             💵 Total PnL: `€{total_pnl:+.2}`\n\n\
             ⚙️ *Config:*\n\
             ⏱ News: every {news_min}min | Housekeeping: every {hk_min}min\n\
             🎯 Max {max_sig} signals/day | Kelly: {kelly:.0}%\n\
             🔍 Min edge: {edge:.0}% | Min volume: ${vol:.0}\n\
             🧠 Model: `{model}` ({agents}-agent consensus)\n\
             🎭 Strategies: {strat_info}",
            news_min = cfg.news_scan_interval_mins,
            hk_min = cfg.scan_interval_mins,
            max_sig = cfg.max_signals_per_day,
            kelly = cfg.kelly_fraction * 100.0,
            edge = cfg.min_effective_edge * 100.0,
            vol = cfg.min_volume,
            model = cfg.llm_model,
            agents = cfg.consensus_agents,
            strat_info = strategies
                .iter()
                .map(|s| format!("{} {}", s.label(), s.name))
                .collect::<Vec<_>>()
                .join(", "),
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
    let housekeeping = tokio::spawn(async move {
        loop {
            if let Err(e) = housekeeping_cycle(&hk_portfolio, &hk_notifier, &hk_scanner).await {
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

    // Spawn heartbeat loop
    let hb_interval = cfg.heartbeat_interval_mins;
    let hb_portfolio = Arc::clone(&portfolio);
    let hb_notifier = Arc::clone(&notifier);
    let hb_cfg = Arc::clone(&cfg);
    let hb_stats = Arc::clone(&stats);
    let heartbeat = tokio::spawn(async move {
        if hb_interval == 0 {
            // Disabled — park forever
            std::future::pending::<()>().await;
            return;
        }
        loop {
            tokio::time::sleep(Duration::from_secs(hb_interval * 60)).await;
            if let Err(e) = heartbeat_cycle(&hb_portfolio, &hb_notifier, &hb_cfg, &hb_stats).await {
                tracing::error!(err = %e, "Heartbeat failed");
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
    }

    Ok(())
}

/// Housekeeping: resolve bets, daily reports, reset counters.
async fn housekeeping_cycle(
    portfolio: &PgPortfolio,
    notifier: &telegram::notifier::TelegramNotifier,
    scanner: &scanner::live::LiveScanner,
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
                    let _ = notifier.send(&msg).await;
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
        let _ = notifier.send(&report).await;
        portfolio.mark_daily_report_sent().await?;
        tracing::info!("Daily report sent");
    }

    // Reload calibration curve with any newly resolved data
    if let Err(e) = scanner.reload_calibration().await {
        tracing::warn!(err = %e, "Failed to reload calibration curve");
    }

    let open_count = portfolio.open_bets().await?.len();
    tracing::info!(open_bets = open_count, "Housekeeping cycle complete");
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

    let skip_ids = portfolio.open_bet_market_ids().await?;
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

            for signal in &result.signals {
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
                    };

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

                            let msg = format!(
                                "{label} *{strat_name}*\n\
                                 {signal}\n\
                                 {news}\n\
                                 💸 *Paper bet placed*\n\
                                 💵 Stake: `€{cost:.2}` ({shares:.1} shares @ `{price:.1}¢`)\n\
                                 🏷 Fees: `€{fee:.2}` (slippage + trading)\n\
                                 💰 Strategy bankroll: `€{bankroll:.2}`\n\
                                 📊 Open bets: {open} | Strategy signals: {today}/{max}",
                                label = strat.label(),
                                strat_name = strat.name,
                                signal = signal.to_telegram_message(),
                                news = news_section,
                                cost = bet_amount,
                                shares = shares,
                                price = slipped_price * 100.0,
                                fee = fee,
                                bankroll = new_strat_bankroll,
                                open = open_count,
                                today = strat_signals,
                                max = strat.max_signals_per_day,
                            );
                            let _ = notifier.send(&msg).await;
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
) -> Result<()> {
    let bankroll = portfolio.bankroll().await?;
    let open_count = portfolio.open_bets().await?.len();
    let signals_today = portfolio.signals_sent_today().await?;

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
         💰 Bankroll: `€{bankroll:.2}`\n\
         📊 Open bets: {open} | Signals today: {today}/{max}",
        interval = cfg.heartbeat_interval_mins,
        open = open_count,
        today = signals_today,
        max = cfg.max_signals_per_day,
    );

    let _ = notifier.send(&msg).await;
    tracing::info!("Heartbeat sent");
    Ok(())
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}...")
    }
}
