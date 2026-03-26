use anyhow::{Context, Result};
use std::time::Duration;

use crate::data::models::GammaMarket;
use crate::live::broadcast;
use crate::storage::portfolio::BetSide;
use crate::storage::postgres::PgPortfolio;
use crate::telegram::notifier::TelegramNotifier;

const GAMMA_API: &str = "https://gamma-api.polymarket.com";

pub async fn housekeeping_cycle(
    portfolio: &PgPortfolio,
    notifier: &TelegramNotifier,
    http: &reqwest::Client,
) -> Result<()> {
    let open_ids = portfolio.open_copy_bet_market_ids().await?;

    for market_id in &open_ids {
        tokio::time::sleep(Duration::from_millis(200)).await;
        match check_market_resolution(http, market_id).await {
            Ok(Some(yes_won)) => {
                if let Some(r) = portfolio.resolve_bet(market_id, yes_won).await? {
                    let emoji = if r.won { "✅" } else { "❌" };
                    let result_label = if r.won { "WON" } else { "LOST" };
                    let side_emoji = match r.side {
                        BetSide::Yes => "🟢 YES",
                        BetSide::No => "🔴 NO",
                    };
                    let roi = bet_roi(r.pnl, r.cost, r.entry_fee);
                    let msg = format!(
                        "{emoji} *Copy Bet {result_label}* ({strat})\n\n\
                         📋 _{question}_\n\
                         🎲 Side: *{side}* — {shares:.1} shares @ `{price:.1}¢`\n\
                         💵 Stake: `€{cost:.2}` → PnL: `€{pnl:+.2}` ({roi:+.0}%)\n\n\
                         💰 Strategy bankroll: `€{bankroll:.2}`\n\
                         📊 Strategy: {sw}W/{sl}L `€{sp:+.2}` | All: {wins}W/{losses}L `€{total_pnl:+.2}`",
                        strat = r.strategy,
                        question = r.question,
                        side = side_emoji,
                        shares = r.shares,
                        price = r.entry_price * 100.0,
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
                    tracing::info!(
                        market = %market_id,
                        strategy = %r.strategy,
                        result = result_label,
                        pnl = format_args!("€{:+.2}", r.pnl),
                        bankroll = format_args!("€{:.2}", r.bankroll),
                        "Copy bet resolved"
                    );
                }
            }
            Ok(None) => {} // still open
            Err(e) => {
                tracing::warn!(market = %market_id, err = %e, "Copy resolution check failed");
            }
        }
    }

    tracing::info!(
        open_copy_bets = open_ids.len(),
        "Copy housekeeping cycle complete"
    );
    Ok(())
}

async fn check_market_resolution(http: &reqwest::Client, market_id: &str) -> Result<Option<bool>> {
    let url = format!("{GAMMA_API}/markets/{market_id}");
    let resp = http.get(&url).send().await?;
    let text = resp.text().await?;
    let market: GammaMarket = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse market {market_id}"))?;
    Ok(market.resolved_yes())
}

fn bet_roi(pnl: f64, cost: f64, entry_fee: f64) -> f64 {
    let total_invested = cost + entry_fee;
    if total_invested > 0.0 {
        pnl / total_invested * 100.0
    } else {
        0.0
    }
}
