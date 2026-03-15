use anyhow::Result;

use crate::config::AppConfig;
use crate::data::models::fetch_yes_prices;
use crate::format;
use crate::live::broadcast;
use crate::metrics;
use crate::pricing::kelly::fractional_kelly;
use crate::scanner::copy_trader::{CopyTraderMonitor, fetch_trader_username};
use crate::scanner::live::{LiveScanner, SignalSource};
use crate::storage::portfolio::{BetSide, NewBet};
use crate::storage::postgres::PgPortfolio;
use crate::telegram::notifier::TelegramNotifier;

/// Default bankroll for a newly followed trader.
pub const COPY_TRADER_STARTING_BANKROLL: f64 = 1000.0;
/// Kelly fraction multiplier (quarter-Kelly for safety).
const KELLY_FRACTION: f64 = 0.25;
/// Minimum bet size.
const MIN_BET: f64 = 3.0;

pub async fn copy_trade_cycle(
    portfolio: &PgPortfolio,
    notifier: &TelegramNotifier,
    scanner: &LiveScanner,
    monitor: &CopyTraderMonitor,
    cfg: &AppConfig,
) -> Result<()> {
    let detected = monitor.detect_new_trades(portfolio).await?;
    if detected.is_empty() {
        return Ok(());
    }

    tracing::info!(count = detected.len(), "Processing copy-trade signals");

    // Skip sets to avoid duplicate bets on same market/event
    let skip_ids = portfolio.open_bet_market_ids().await?;
    let skip_event_slugs = portfolio.open_bet_event_slugs().await?;

    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    // Backfill usernames for any traders that don't have one yet
    if let Ok(traders) = portfolio.get_active_traders().await {
        for trader in traders.iter().filter(|t| t.username.is_none()) {
            if let Some(name) = fetch_trader_username(&http, &trader.proxy_wallet).await {
                tracing::info!(wallet = %trader.proxy_wallet, username = %name, "Backfilled trader username");
                let _ = portfolio
                    .update_trader_username(&trader.proxy_wallet, &name)
                    .await;
            }
        }
    }

    let mut bets_placed = 0usize;
    let mut exits_triggered = 0usize;

    for dt in &detected {
        let trade = &dt.trade;

        let wallet_short = &dt.trader_wallet[..8.min(dt.trader_wallet.len())];
        let strategy_name = format!("copy:{wallet_short}");

        // Mirror exits: if the trader sold a position we copied, exit ours too.
        if trade.side == "SELL" {
            // Resolve slug → Gamma numeric ID to match against bets table
            let gamma_id = match scanner.fetch_market_by_slug(&trade.slug).await {
                Ok(m) => m.market_id,
                Err(e) => {
                    tracing::warn!(slug = %trade.slug, err = %e, "Copy-exit: failed to resolve market slug");
                    continue;
                }
            };
            let open_bets = portfolio.open_bets().await?;
            let matching = open_bets
                .iter()
                .find(|b| b.market_id == gamma_id && b.strategy == strategy_name);

            if let Some(bet) = matching {
                let ids = [bet.market_id.as_str()];
                let yes_price = fetch_yes_prices(&http, &ids)
                    .await
                    .into_iter()
                    .next()
                    .flatten();

                let Some(current_yes_price) = yes_price else {
                    tracing::warn!(
                        market = %gamma_id,
                        trader = wallet_short,
                        "Copy-exit: could not fetch current price, skipping"
                    );
                    continue;
                };

                let trader = portfolio
                    .get_trader_by_wallet(&dt.trader_wallet)
                    .await
                    .ok()
                    .flatten();
                let trader_name = trader
                    .as_ref()
                    .and_then(|t| t.username.as_deref())
                    .unwrap_or(wallet_short);

                let reason = format!(
                    "copy-exit: {trader_name} sold at {:.1}¢",
                    trade.price * 100.0
                );

                match portfolio
                    .early_exit(bet.id, current_yes_price, &reason)
                    .await
                {
                    Ok(Some(r)) => {
                        let side_emoji = match r.side {
                            BetSide::Yes => "🟢 YES",
                            BetSide::No => "🔴 NO",
                        };
                        let msg = format!(
                            "👥 *Copy Exit*\n\
                             📋 _{question}_\n\
                             🎲 Side: *{side}* — {shares:.1} shares @ `{entry:.1}¢` → `{now:.1}¢`\n\
                             👤 Trader: `{trader}` sold at `{sold:.1}¢`\n\
                             💵 PnL: `€{pnl:+.2}`\n\
                             💰 Trader bankroll: `€{bankroll:.2}`",
                            question = format::truncate(&r.question, 60),
                            side = side_emoji,
                            shares = r.shares,
                            entry = r.entry_price * 100.0,
                            now = current_yes_price * 100.0,
                            trader = trader_name,
                            sold = trade.price * 100.0,
                            pnl = r.pnl,
                            bankroll = r.bankroll,
                        );
                        broadcast(notifier, portfolio, &msg).await;
                        tracing::info!(
                            market = %gamma_id,
                            trader = %trader_name,
                            pnl = format_args!("€{:+.2}", r.pnl),
                            "Copy-exit executed"
                        );
                        exits_triggered += 1;
                    }
                    Ok(None) => {}
                    Err(e) => tracing::error!(err = %e, "Copy-exit early_exit failed"),
                }
            } else {
                tracing::debug!(
                    slug = %trade.slug,
                    trader = wallet_short,
                    "Copy-exit: no matching open bet, ignoring SELL"
                );
            }
            continue;
        }

        // BUY path below — only copy entries
        if trade.side != "BUY" {
            continue;
        }

        // Fetch market data (resolves slug → Gamma numeric ID + events)
        let market = match scanner.fetch_market_by_slug(&trade.slug).await {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(slug = %trade.slug, err = %e, "Failed to fetch market for copy-trade");
                continue;
            }
        };

        // Skip if we already have an open bet on this market (use Gamma numeric ID)
        if skip_ids.contains(&market.market_id) {
            tracing::debug!(market = %market.market_id, "Copy-trade skip: already have open bet");
            continue;
        }

        // Skip if same event already has an open bet
        if let Some(slug) = market.event_slug()
            && skip_event_slugs.iter().any(|s| s == slug)
        {
            tracing::debug!(
                event_slug = slug,
                "Copy-trade skip: same event already bet on"
            );
            continue;
        }

        // Get trader's dedicated bankroll
        let trader_bankroll = portfolio.strategy_bankroll(&strategy_name).await?;
        if trader_bankroll < MIN_BET {
            tracing::debug!(
                trader = wallet_short,
                bankroll = trader_bankroll,
                "Copy-trade skip: insufficient trader bankroll"
            );
            continue;
        }

        // Derive probability estimate: trader's entry price + 5% markup
        let entry_price = trade.price;
        let estimated_prob = (entry_price + 0.05).min(0.95);

        // Kelly sizing
        let kelly = fractional_kelly(estimated_prob, entry_price, KELLY_FRACTION);
        if kelly <= 0.0 {
            continue;
        }

        let raw_bet = trader_bankroll * kelly;
        if raw_bet < MIN_BET {
            tracing::debug!(
                trader = wallet_short,
                kelly_bet = raw_bet,
                "Copy-trade skip: kelly bet below minimum"
            );
            continue;
        }

        let bet_amount = raw_bet;
        let slipped_price = (entry_price * (1.0 + cfg.slippage_pct)).min(0.99);
        let shares = bet_amount / slipped_price;
        let fee = bet_amount * cfg.fee_pct;
        let total_cost = bet_amount + fee;

        if total_cost > trader_bankroll {
            tracing::debug!(
                trader = wallet_short,
                cost = total_cost,
                bankroll = trader_bankroll,
                "Copy-trade skip: cost exceeds bankroll"
            );
            continue;
        }

        // Run ML model for informational purposes (not a gate)
        let ml_info = match scanner.predict_market(&market.market_id, entry_price).await {
            Some((ml_prob, ml_conf)) => {
                let agrees = (ml_prob > 0.5 && entry_price < 0.5)
                    || (ml_prob < 0.5 && entry_price > 0.5)
                    || (ml_prob > entry_price);
                let direction = if agrees { "agrees" } else { "disagrees" };
                format!(
                    "🤖 Model: {:.1}% (conf {:.0}%) — {direction}",
                    ml_prob * 100.0,
                    ml_conf * 100.0,
                )
            }
            None => "🤖 Model: unavailable".to_string(),
        };

        // Get trader info for the message
        let trader = portfolio
            .get_trader_by_wallet(&dt.trader_wallet)
            .await
            .ok()
            .flatten();
        let trader_display = if let Some(ref t) = trader {
            let name = t.username.as_deref().unwrap_or(wallet_short);
            let rank = t.rank.map(|r| format!(" | Rank #{r}")).unwrap_or_default();
            let pnl = t
                .pnl
                .map(|p| format!(" | PnL ${:.0}k", p / 1000.0))
                .unwrap_or_default();
            format!("{name}{rank}{pnl}")
        } else {
            wallet_short.to_string()
        };

        let edge = estimated_prob - entry_price;
        let reasoning = format!(
            "Copy-trade: {} bought at {:.1}%",
            trader_display,
            entry_price * 100.0,
        );

        let new_bet = NewBet {
            market_id: market.market_id.clone(),
            question: market.question.clone(),
            side: BetSide::Yes,
            entry_price,
            slipped_price,
            shares,
            cost: bet_amount,
            fee,
            estimated_prob,
            confidence: 0.50,
            edge,
            kelly_size: kelly,
            reasoning,
            end_date: market.end_date.clone(),
            context: None,
            strategy: strategy_name.clone(),
            source: SignalSource::CopyTrade.as_str().to_string(),
            url: market.polymarket_url(),
            event_slug: market.event_slug().map(String::from),
        };

        // Log prediction for model learning (Brier score tracking)
        let _ = portfolio
            .log_prediction(
                &market.market_id,
                SignalSource::CopyTrade.as_str(),
                entry_price,
                estimated_prob,
                estimated_prob,
                0.50,
                edge,
            )
            .await;

        match portfolio.place_bet(&new_bet).await {
            Ok(_bet_id) => {
                let new_bankroll = portfolio.strategy_bankroll(&strategy_name).await?;
                let open_count = portfolio.open_bets().await?.len();
                metrics::record_bet(&strategy_name, SignalSource::CopyTrade.as_str(), bet_amount);
                metrics::record_bankroll(&strategy_name, new_bankroll);
                metrics::record_open_bets(open_count as u64);

                // Get trader's record with us
                let trader_record = portfolio.copy_trader_record(&strategy_name).await?;

                let msg = crate::format::format_copy_bet(&crate::format::CopyBetNotif {
                    question: &format::truncate(&market.question, 60),
                    cost: bet_amount,
                    shares,
                    price_cents: slipped_price * 100.0,
                    edge_pct: edge * 100.0,
                    kelly_pct: kelly * 100.0,
                    ml_info: &ml_info,
                    trader_display: &trader_display,
                    wins: trader_record.0,
                    losses: trader_record.1,
                    trader_pnl: trader_record.2,
                    bankroll: new_bankroll,
                    open: open_count,
                });
                broadcast(notifier, portfolio, &msg).await;
                bets_placed += 1;
            }
            Err(e) => {
                tracing::error!(err = %e, "Failed to place copy-trade bet");
            }
        }
    }

    if bets_placed > 0 || exits_triggered > 0 {
        tracing::info!(bets_placed, exits_triggered, "Copy-trade cycle complete");
    }
    Ok(())
}
