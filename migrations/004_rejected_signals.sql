CREATE TABLE IF NOT EXISTS rejected_signals (
    id          SERIAL PRIMARY KEY,
    market_id   TEXT NOT NULL,
    question    TEXT NOT NULL,
    reason      TEXT NOT NULL,
    current_price DOUBLE PRECISION,
    estimated_prob DOUBLE PRECISION,
    edge        DOUBLE PRECISION,
    confidence  DOUBLE PRECISION,
    combined_lr DOUBLE PRECISION,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rejected_signals_created ON rejected_signals (created_at);
