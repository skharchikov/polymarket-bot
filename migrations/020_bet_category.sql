-- ADR 009: Store market category on bets for post-hoc analysis.
-- Previously required regex heuristics on question text.
ALTER TABLE bets ADD COLUMN IF NOT EXISTS category TEXT;
