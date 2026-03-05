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
use std::time::Duration;
use storage::portfolio::PortfolioState;

const MAX_SIGNALS_PER_DAY: usize = 3;
const SCAN_INTERVAL_MINS: u64 = 30;
const STARTING_BANKROLL: f64 = 300.0;

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
    let mut portfolio = PortfolioState::load_or_create(STARTING_BANKROLL);

    let _ = notifier
        .send(&format!(
            "🤖 *Polymarket Signal Bot started*\n\
             💰 Bankroll: `€{:.2}`\n\
             📊 Open bets: {}\n\
             Scanning crypto markets for high-edge YES opportunities...",
            portfolio.bankroll,
            portfolio.open_bets().len(),
        ))
        .await;

    loop {
        portfolio.reset_daily_if_needed();

        // Send daily report if we haven't today
        if portfolio.should_send_daily_report() && !portfolio.bets.is_empty() {
            portfolio.take_snapshot();
            let report = portfolio.daily_summary();
            let _ = notifier.send(&report).await;
            portfolio.mark_daily_report_sent();
            tracing::info!("Daily report sent");
        }

        let remaining = MAX_SIGNALS_PER_DAY.saturating_sub(portfolio.signals_sent_today);
        if remaining == 0 {
            tracing::info!("Daily signal limit reached");
            tokio::time::sleep(Duration::from_secs(SCAN_INTERVAL_MINS * 60)).await;
            continue;
        }

        tracing::info!(
            remaining = remaining,
            bankroll = format_args!("€{:.2}", portfolio.bankroll),
            "Starting scan..."
        );

        // Skip markets we already have open bets on
        let skip_ids: Vec<String> = portfolio
            .open_bets()
            .iter()
            .map(|b| b.market_id.clone())
            .collect();

        match scanner.scan(&skip_ids).await {
            Ok(signals) => {
                for signal in signals.into_iter().take(remaining) {
                    // Place paper bet with realistic costs
                    if let Some(bet) = portfolio.place_bet(
                        &signal.market_id,
                        &signal.question,
                        signal.current_price,
                        signal.kelly_size,
                        signal.estimated_prob,
                        signal.confidence,
                        signal.edge,
                        &signal.reasoning,
                        signal.end_date.as_deref(),
                    ) {
                        tracing::info!(
                            market = %bet.question,
                            cost = format_args!("€{:.2}", bet.cost),
                            price = format_args!("{:.1}¢", bet.slipped_price * 100.0),
                            edge = format_args!("+{:.1}%", bet.edge * 100.0),
                            bankroll = format_args!("€{:.2}", portfolio.bankroll),
                            "Bet placed"
                        );

                        // Send signal + bet info to Telegram
                        let msg = format!(
                            "{}\n\n💸 *Paper bet placed:* `€{:.2}` @ `{:.1}¢` (incl. {:.1}% slippage + {:.1}% fee)\n\
                             💰 Remaining bankroll: `€{:.2}`",
                            signal.to_telegram_message(),
                            bet.cost,
                            bet.slipped_price * 100.0,
                            1.0, // slippage %
                            2.0, // fee %
                            portfolio.bankroll,
                        );
                        let _ = notifier.send(&msg).await;
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
            signals_today = portfolio.signals_sent_today,
            open_bets = portfolio.open_bets().len(),
            "Scan cycle complete"
        );
        tokio::time::sleep(Duration::from_secs(SCAN_INTERVAL_MINS * 60)).await;
    }
}
