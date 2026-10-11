-- Execution mode per bet. Paper and live figures must never be mixed.
ALTER TABLE bets ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'paper';
ALTER TABLE bets DROP CONSTRAINT IF EXISTS bets_mode_check;
ALTER TABLE bets ADD CONSTRAINT bets_mode_check CHECK (mode IN ('paper', 'shadow', 'live'));
CREATE INDEX IF NOT EXISTS idx_bets_mode_resolved ON bets (mode, resolved);
