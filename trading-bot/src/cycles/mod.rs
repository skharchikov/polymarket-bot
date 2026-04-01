mod alerts;
mod bet_scan;
mod heartbeat;
mod housekeeping;

pub use alerts::alert_loop;
pub use bet_scan::bet_scan_cycle;
pub use heartbeat::heartbeat_cycle;
pub use housekeeping::housekeeping_cycle;

use crate::scanner::live::{CorrelationDecision, LiveScanner};
use crate::storage::portfolio::BetSide;
use crate::storage::postgres::PgPortfolio;

/// Runs the LLM portfolio correlation check for a slice of candidates against
/// currently open bets and recently blocked signals. Returns one decision per
/// candidate, defaulting to `keep = true` on any error or when there is nothing
/// to correlate against.
pub(super) async fn portfolio_correlation_check(
    scanner: &LiveScanner,
    portfolio: &PgPortfolio,
    candidates: &[(String, String, BetSide)],
) -> Vec<CorrelationDecision> {
    let open_bets = portfolio.open_ml_bets().await.unwrap_or_default();
    let blocked = portfolio
        .recent_correlation_blocked()
        .await
        .unwrap_or_default();
    if candidates.is_empty() || (open_bets.is_empty() && blocked.is_empty()) {
        return candidates
            .iter()
            .map(|_| CorrelationDecision {
                keep: true,
                reason: "No open bets to correlate against".to_string(),
            })
            .collect();
    }
    tracing::info!(
        candidates = candidates.len(),
        open_bets = open_bets.len(),
        blocked = blocked.len(),
        "Running portfolio correlation check"
    );
    scanner
        .check_portfolio_correlation(candidates, &open_bets, &blocked)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(err = %e, "Correlation check failed, keeping all candidates");
            candidates
                .iter()
                .map(|_| CorrelationDecision {
                    keep: true,
                    reason: "Check failed — defaulting to keep".to_string(),
                })
                .collect()
        })
}
