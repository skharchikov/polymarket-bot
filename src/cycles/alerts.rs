use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{RwLock, mpsc};

use crate::config::AppConfig;
use crate::format;
use crate::live::broadcast;
use crate::metrics;
use crate::scanner::live::LiveScanner;
use crate::scanner::ws::ActivityAlert;
use crate::storage::portfolio::{BetSide, NewBet};
use crate::storage::postgres::PgPortfolio;
use crate::strategy::StrategyProfile;
use crate::telegram::notifier::TelegramNotifier;

pub async fn alert_loop(
    mut alert_rx: mpsc::Receiver<ActivityAlert>,
    scanner: Arc<LiveScanner>,
    portfolio: Arc<PgPortfolio>,
    notifier: Arc<TelegramNotifier>,
    strategies: Arc<Vec<StrategyProfile>>,
    cfg: Arc<AppConfig>,
    token_map: Arc<RwLock<HashMap<String, String>>>,
) {
    // Throttle: don't re-assess same market within 15 minutes
    let mut last_assessed: HashMap<String, std::time::Instant> = HashMap::new();
    // Global WS cooldown: max 1 WS-triggered bet per 10 minutes
    let mut last_ws_bet = std::time::Instant::now() - Duration::from_secs(600);
    // Max WS bets per day
    let mut ws_bets_today: usize = 0;
    let mut ws_bets_date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    const MAX_WS_BETS_PER_DAY: usize = 3;
    // Cooldown for open-bet price move notifications (1h per market)
    let mut last_price_alert: HashMap<String, std::time::Instant> = HashMap::new();

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

        let map = token_map.read().await;
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

        // If we already have an open bet, notify about the move but don't bet again
        let open_bets = portfolio.open_bets().await.unwrap_or_default();
        let open_bet = open_bets.iter().find(|b| b.market_id == market_id);
        if let Some(bet) = open_bet {
            let delta_pct = (alert.price - alert.prev_price) * 100.0;
            // Only alert on 5%+ moves with 1h cooldown per market
            if delta_pct.abs() < 5.0 {
                continue;
            }
            if let Some(last) = last_price_alert.get(&market_id)
                && now.duration_since(*last) < Duration::from_secs(3600)
            {
                continue;
            }
            last_price_alert.insert(market_id.clone(), now);

            let trade_info = alert
                .trade_size
                .map(|s| format!(" | Trade: `${s:.0}`"))
                .unwrap_or_default();
            let arrow = if delta_pct > 0.0 { "📈" } else { "📉" };
            let q = format::truncate(&bet.question, 50);
            let msg = format!(
                "{arrow} *Price Move* on open bet\n\n\
                 📋 _{q}_\n\
                 💰 `{prev:.1}¢` → `{now:.1}¢` ({delta:+.1}%){trade}",
                prev = alert.prev_price * 100.0,
                now = alert.price * 100.0,
                delta = delta_pct,
                trade = trade_info,
            );
            let _ = notifier.send(&msg).await;
            continue;
        }

        tracing::info!(
            market_id = &market_id[..16.min(market_id.len())],
            price = format_args!("{:.1}%", alert.price * 100.0),
            delta = format_args!("{:+.1}%", (alert.price - alert.prev_price) * 100.0),
            "WS alert → assessing market"
        );

        match scanner.assess_alert(&market_id, alert.price).await {
            Ok(Some(signal)) => {
                metrics::record_ws_alert(true);
                tracing::info!(
                    market = %signal.question,
                    side = %signal.side,
                    edge = format_args!("+{:.1}%", signal.edge * 100.0),
                    "WS-triggered signal found"
                );

                // Skip if another market in the same event already has an open bet
                if let Some(ref slug) = signal.event_slug {
                    let open_slugs = portfolio.open_bet_event_slugs().await.unwrap_or_default();
                    if open_slugs.iter().any(|s| s == slug) {
                        tracing::info!(
                            event_slug = slug,
                            "Skipping WS signal — already have bet on this event"
                        );
                        continue;
                    }
                }

                // Process through strategies — only first matching strategy bets
                for strat in strategies.iter() {
                    let sent = portfolio
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

                    let strat_bankroll = portfolio
                        .strategy_bankroll(&strat.name)
                        .await
                        .unwrap_or(0.0);
                    let raw_bet = strat_bankroll * accepted.kelly_size;
                    if raw_bet < strat.min_bet {
                        continue;
                    }

                    let slipped_price = (signal.current_price * (1.0 + cfg.slippage_pct)).min(0.99);
                    let shares = raw_bet / slipped_price;
                    let fee = raw_bet * cfg.fee_pct;
                    let total_cost = raw_bet + fee;
                    if total_cost > strat_bankroll {
                        continue;
                    }

                    let source_str = signal.source.as_str();
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
                        url: signal.polymarket_url.clone(),
                        event_slug: signal.event_slug.clone(),
                    };

                    // Log prediction for Brier score tracking
                    let _ = portfolio
                        .log_prediction(
                            &signal.market_id,
                            source_str,
                            signal.prior,
                            signal.estimated_prob,
                            signal.estimated_prob,
                            signal.confidence,
                            signal.edge,
                        )
                        .await;

                    match portfolio.place_bet(&new_bet).await {
                        Ok(_) => {
                            last_ws_bet = std::time::Instant::now();
                            ws_bets_today += 1;
                            let new_bankroll = portfolio
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
                            broadcast(&notifier, &portfolio, &msg).await;
                            break;
                        }
                        Err(e) => {
                            tracing::error!(err = %e, "Failed to place WS-triggered bet");
                        }
                    }
                }
            }
            Ok(None) => {
                metrics::record_ws_alert(false);
                tracing::debug!(
                    market_id = &market_id[..16.min(market_id.len())],
                    "WS alert — no edge"
                );
            }
            Err(e) => {
                metrics::record_ws_alert(false);
                tracing::warn!(err = %e, "WS alert assessment failed");
            }
        }
    }
}
