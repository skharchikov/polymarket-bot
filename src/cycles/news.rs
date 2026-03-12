use anyhow::Result;
use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::config::AppConfig;
use crate::format;
use crate::live::{ScanStats, broadcast};
use crate::metrics;
use crate::scanner::live::LiveScanner;
use crate::storage::portfolio::{BetSide, NewBet};
use crate::storage::postgres::PgPortfolio;
use crate::strategy::StrategyProfile;
use crate::telegram::notifier::TelegramNotifier;

pub async fn news_scan_cycle(
    portfolio: &PgPortfolio,
    notifier: &TelegramNotifier,
    scanner: &LiveScanner,
    cfg: &AppConfig,
    stats: &ScanStats,
    strategies: &[StrategyProfile],
    seen_headlines: &mut HashSet<String>,
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
    // Skip markets belonging to events we already have open bets on
    let skip_event_slugs = portfolio.open_bet_event_slugs().await?;
    let past_bets = portfolio.learning_summary().await?;

    match scanner
        .scan(&skip_ids, &skip_event_slugs, &past_bets, seen_headlines)
        .await
    {
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

            metrics::record_scan(
                result.markets_scanned as u64,
                result.news_total as u64,
                result.news_new as u64,
                result.signals.len() as u64,
            );

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
                                format::truncate(&signal.question, 40),
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
                            format::truncate(&signal.question, 40),
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
                            format::truncate(&signal.question, 40),
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
                        source: signal.source.as_str().to_string(),
                        url: signal.polymarket_url.clone(),
                        event_slug: signal.event_slug.clone(),
                    };

                    // Log prediction for Brier score tracking
                    let _ = portfolio
                        .log_prediction(
                            &signal.market_id,
                            signal.source.as_str(),
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
                            metrics::record_bet(&strat.name, signal.source.as_str(), bet_amount);
                            metrics::record_bankroll(&strat.name, new_strat_bankroll);
                            metrics::record_open_bets(open_count as u64);
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
                                format::truncate(&signal.question, 40),
                                bet_amount,
                            ));

                            let news_section = if !signal.context.news_headlines.is_empty() {
                                let headlines: Vec<String> = signal
                                    .context
                                    .news_headlines
                                    .iter()
                                    .take(3)
                                    .map(|h| format!("  • _{}_", format::truncate(h, 80)))
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
