CREATE TABLE IF NOT EXISTS correlation_blocked (
    id         SERIAL PRIMARY KEY,
    market_id  TEXT NOT NULL,
    question   TEXT NOT NULL,
    side       TEXT NOT NULL,
    reason     TEXT NOT NULL,
    end_date   TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_correlation_blocked_end_date ON correlation_blocked (end_date);
CREATE INDEX IF NOT EXISTS idx_correlation_blocked_created  ON correlation_blocked (created_at);
