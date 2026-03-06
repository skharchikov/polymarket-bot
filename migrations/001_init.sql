CREATE TABLE IF NOT EXISTS portfolio (
    id          SERIAL PRIMARY KEY,
    key         TEXT UNIQUE NOT NULL,
    value_f64   DOUBLE PRECISION,
    value_text  TEXT
);

-- Seed portfolio meta rows
INSERT INTO portfolio (key, value_f64) VALUES ('starting_bankroll', 300.0) ON CONFLICT (key) DO NOTHING;
INSERT INTO portfolio (key, value_f64) VALUES ('bankroll', 300.0) ON CONFLICT (key) DO NOTHING;
INSERT INTO portfolio (key, value_f64) VALUES ('signals_sent_today', 0.0) ON CONFLICT (key) DO NOTHING;
INSERT INTO portfolio (key, value_text) VALUES ('last_signal_date', '') ON CONFLICT (key) DO NOTHING;
INSERT INTO portfolio (key, value_text) VALUES ('last_daily_report_date', '') ON CONFLICT (key) DO NOTHING;

CREATE TABLE IF NOT EXISTS bets (
    id              SERIAL PRIMARY KEY,
    market_id       TEXT NOT NULL,
    question        TEXT NOT NULL,
    side            TEXT NOT NULL DEFAULT 'Yes',
    entry_price     DOUBLE PRECISION NOT NULL,
    slipped_price   DOUBLE PRECISION NOT NULL,
    shares          DOUBLE PRECISION NOT NULL,
    cost            DOUBLE PRECISION NOT NULL,
    fee_paid        DOUBLE PRECISION NOT NULL,
    estimated_prob  DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    edge            DOUBLE PRECISION NOT NULL,
    kelly_size      DOUBLE PRECISION NOT NULL,
    reasoning       TEXT NOT NULL,
    end_date        TEXT,
    context         JSONB,
    placed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved        BOOLEAN NOT NULL DEFAULT FALSE,
    won             BOOLEAN,
    pnl             DOUBLE PRECISION,
    resolved_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_bets_market_id ON bets(market_id);
CREATE INDEX IF NOT EXISTS idx_bets_resolved ON bets(resolved);

CREATE TABLE IF NOT EXISTS daily_snapshots (
    id          SERIAL PRIMARY KEY,
    date        TEXT NOT NULL,
    bankroll    DOUBLE PRECISION NOT NULL,
    open_bets   INTEGER NOT NULL,
    total_bets  INTEGER NOT NULL,
    wins        INTEGER NOT NULL,
    losses      INTEGER NOT NULL,
    total_pnl   DOUBLE PRECISION NOT NULL,
    roi_pct     DOUBLE PRECISION NOT NULL
);
