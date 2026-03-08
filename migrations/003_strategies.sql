ALTER TABLE bets ADD COLUMN IF NOT EXISTS strategy TEXT NOT NULL DEFAULT 'balanced';
CREATE INDEX IF NOT EXISTS idx_bets_strategy ON bets(strategy);
