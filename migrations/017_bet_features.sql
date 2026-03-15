-- Store exact feature vector at bet-placement time for online learning.
-- JSONB chosen over DOUBLE PRECISION[] for extensibility: missing keys default to 0.0,
-- new features don't invalidate old records, self-describing for Python training pipeline.
CREATE TABLE IF NOT EXISTS bet_features (
    bet_id      INTEGER PRIMARY KEY REFERENCES bets(id) ON DELETE CASCADE,
    features    JSONB    NOT NULL,
    version     INTEGER  NOT NULL DEFAULT 1,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
