-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_bets_strategy_resolved ON bets(strategy, resolved);
CREATE INDEX IF NOT EXISTS idx_bets_resolved_placed ON bets(resolved, placed_at DESC);
