use anyhow::Result;
use std::time::Duration;

use crate::format;
use crate::live::{broadcast, notify_owner, random_victory_gif, should_send_gif};
use crate::metrics;
use crate::scanner::live::LiveScanner;
use crate::storage::portfolio::BetSide;
use crate::storage::postgres::PgPortfolio;
use crate::strategy;
use crate::telegram::notifier::TelegramNotifier;

pub async fn housekeeping_cycle(
    portfolio: &PgPortfolio,
    notifier: &TelegramNotifier,
    scanner: &LiveScanner,
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
                    let strat_label = strategy::strategy_label(&r.strategy);
                    let src_icon = strategy::source_icon(&r.source);
                    let side_emoji = match r.side {
                        BetSide::Yes => "🟢 YES",
                        BetSide::No => "🔴 NO",
                    };
                    let msg = format!(
                        "{emoji} *Bet {result_label}* ({strat_label} {strat})\n\n\
                         📋 _{question}_\n\
                         🎲 Side: *{side}* — {shares:.1} shares @ `{price:.1}¢`\n\
                         {src_icon} Edge: `+{edge:.1}%` | Conf: `{conf:.0}%`\n\
                         💵 Stake: `€{cost:.2}` → PnL: `€{pnl:+.2}` ({roi:+.0}%)\n\n\
                         💰 Strategy bankroll: `€{bankroll:.2}`\n\
                         📊 Strategy: {sw}W/{sl}L `€{sp:+.2}` | All: {wins}W/{losses}L `€{total_pnl:+.2}`",
                        strat = r.strategy,
                        question = r.question,
                        side = side_emoji,
                        shares = r.shares,
                        price = r.entry_price * 100.0,
                        edge = r.edge * 100.0,
                        conf = r.confidence * 100.0,
                        cost = r.cost,
                        pnl = r.pnl,
                        bankroll = r.bankroll,
                        sw = r.strat_wins,
                        sl = r.strat_losses,
                        sp = r.strat_pnl,
                        wins = r.total_wins,
                        losses = r.total_losses,
                        total_pnl = r.total_pnl,
                    );
                    broadcast(notifier, portfolio, &msg).await;
                    if r.won && should_send_gif() {
                        let gif = random_victory_gif();
                        let subs = portfolio.telegram_subscribers().await.unwrap_or_default();
                        notifier.broadcast_animation(&subs, gif).await;
                    }
                    metrics::record_resolution(&r.strategy, r.won, r.pnl);
                    metrics::record_bankroll(&r.strategy, r.bankroll);
                    tracing::info!(
                        market = %market_id,
                        strategy = %r.strategy,
                        source = %r.source,
                        result = result_label,
                        pnl = format_args!("€{:+.2}", r.pnl),
                        bankroll = format_args!("€{:.2}", r.bankroll),
                        "Bet resolved"
                    );
                    // Trigger warm-start retrain — sidecar applies its own threshold
                    // (WARMSTART_TRIGGER_N) and skips if not enough resolved bets yet.
                    scanner.trigger_warmstart().await;
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
        let mut views = Vec::new();

        for bet in &open_bets {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let current = match scanner.fetch_current_price(&bet.market_id).await {
                Ok(Some(p)) => p,
                Ok(None) => {
                    views.push(format::OpenBetView {
                        bet,
                        current_yes_price: None,
                        poly_url: None,
                    });
                    continue;
                }
                Err(e) => {
                    tracing::debug!(market = %bet.market_id, err = %e, "Price fetch failed");
                    views.push(format::OpenBetView {
                        bet,
                        current_yes_price: None,
                        poly_url: None,
                    });
                    continue;
                }
            };

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
            } else if exit_days_before_expiry > 0
                && days_left <= exit_days_before_expiry
                && loss_pct >= 0.10
            {
                Some(format!(
                    "expiry exit ({days_left}d left, {:.0}% underwater)",
                    loss_pct * 100.0
                ))
            } else {
                None
            };

            if let Some(reason) = exit_reason {
                match portfolio.early_exit(bet.id, current, &reason).await {
                    Ok(Some(r)) => {
                        let strat_label = strategy::strategy_label(&r.strategy);
                        let side_emoji = match r.side {
                            BetSide::Yes => "🟢 YES",
                            BetSide::No => "🔴 NO",
                        };
                        let msg = format!(
                            "🚨 *Early Exit* ({strat_label} {strat})\n\n\
                             📋 _{question}_\n\
                             🎲 Side: *{side}* — {shares:.1} shares @ `{entry:.1}¢` → `{now:.1}¢`\n\
                             💡 Reason: {reason}\n\
                             💵 PnL: `€{pnl:+.2}`\n\n\
                             💰 Strategy bankroll: `€{bankroll:.2}`\n\
                             📊 Strategy: {sw}W/{sl}L `€{sp:+.2}` | All: {wins}W/{losses}L `€{total_pnl:+.2}`",
                            strat = r.strategy,
                            question = r.question,
                            side = side_emoji,
                            shares = r.shares,
                            entry = r.entry_price * 100.0,
                            now = current * 100.0,
                            pnl = r.pnl,
                            bankroll = r.bankroll,
                            sw = r.strat_wins,
                            sl = r.strat_losses,
                            sp = r.strat_pnl,
                            wins = r.total_wins,
                            losses = r.total_losses,
                            total_pnl = r.total_pnl,
                        );
                        broadcast(notifier, portfolio, &msg).await;
                        tracing::info!(
                            market = %bet.question,
                            strategy = %r.strategy,
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

            views.push(format::OpenBetView {
                bet,
                current_yes_price: Some(current),
                poly_url: None,
            });
        }

        // Cache unrealized PnL for /stats command
        let total_unrealized: f64 = views.iter().filter_map(|v| v.unrealized()).sum();
        let total_cost: f64 = views.iter().map(|v| v.bet.cost).sum();
        let _ = portfolio
            .upsert_f64_pub("unrealized_pnl", total_unrealized)
            .await;
        let _ = portfolio.upsert_f64_pub("open_exposure", total_cost).await;
        metrics::record_unrealized_pnl(total_unrealized);

        let msg = format::format_open_bets(&views, true);
        notify_owner(notifier, &msg).await;
    }

    metrics::record_housekeeping();
    metrics::record_open_bets(open_bets.len() as u64);
    tracing::info!(open_bets = open_bets.len(), "Housekeeping cycle complete");
    Ok(())
}
