use anyhow::Result;
use crate::strategies::mispricing::TradeSignal;

pub struct PolymarketExecutor {
    // TODO: add Polygon provider + contract instances
    dry_run: bool,
}

impl PolymarketExecutor {
    pub fn new(dry_run: bool) -> Self {
        Self { dry_run }
    }

    pub async fn execute(&self, signal: &TradeSignal, amount_usd: f64) -> Result<String> {
        if self.dry_run {
            tracing::info!(
                market = %signal.market_id,
                side = ?signal.side,
                amount = amount_usd,
                edge = signal.edge,
                "DRY RUN: would place order"
            );
            return Ok("dry-run-tx".to_string());
        }

        // TODO: implement actual Polymarket CLOB order placement
        // 1. Build order via Polymarket CLOB API
        // 2. Sign with wallet
        // 3. Submit order
        anyhow::bail!("Live execution not yet implemented")
    }
}
