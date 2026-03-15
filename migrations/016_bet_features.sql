-- Online learning feature store (ADR 004).
-- Captures the exact 15-feature vector at bet-placement time.
-- JSONB chosen over DOUBLE PRECISION[] for extensibility:
--   adding/removing features does not break old records.
CREATE TABLE bet_features (
    bet_id      INTEGER PRIMARY KEY REFERENCES bets(id) ON DELETE CASCADE,
    features    JSONB       NOT NULL,
    version     INTEGER     NOT NULL DEFAULT 1,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
