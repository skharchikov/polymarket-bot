ALTER TABLE bets ADD COLUMN IF NOT EXISTS event_slug TEXT;
CREATE INDEX IF NOT EXISTS idx_bets_event_slug ON bets(event_slug);
