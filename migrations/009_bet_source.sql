-- Track signal source (xgboost, llm_consensus) for per-source stats.
ALTER TABLE bets ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'unknown';
CREATE INDEX IF NOT EXISTS idx_bets_source ON bets(source);
