use crate::strategies::signal::Signal;
use anyhow::Result;

pub struct PolymarketExecutor {
    dry_run: bool,
}

impl PolymarketExecutor {
    pub fn new(dry_run: bool) -> Self {
        Self { dry_run }
    }

    pub async fn execute(&self, signal: &Signal, amount_usd: f64) -> Result<String> {
        if self.dry_run {
            tracing::info!(
                market = %signal.market_id,
                side = ?signal.side,
                amount = amount_usd,
                edge = signal.edge,
                source = ?signal.source,
                "DRY RUN: would place order"
            );
            return Ok("dry-run-tx".to_string());
        }

        // TODO: implement actual Polymarket CLOB order placement
        anyhow::bail!("Live execution not yet implemented")
    }
}
