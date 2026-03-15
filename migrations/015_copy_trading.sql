CREATE TABLE IF NOT EXISTS followed_traders (
    id              SERIAL PRIMARY KEY,
    proxy_wallet    TEXT UNIQUE NOT NULL,
    username        TEXT,
    source          TEXT NOT NULL DEFAULT 'leaderboard',
    rank            INTEGER,
    pnl             DOUBLE PRECISION,
    volume          DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_checked_at TIMESTAMPTZ,
    active          BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS copy_trade_events (
    id              SERIAL PRIMARY KEY,
    trader_wallet   TEXT NOT NULL,
    market_id       TEXT NOT NULL,
    condition_id    TEXT NOT NULL,
    side            TEXT NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    size_usd        DOUBLE PRECISION NOT NULL,
    tx_hash         TEXT,
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acted_on        BOOLEAN NOT NULL DEFAULT FALSE,
    skip_reason     TEXT
);

CREATE INDEX idx_copy_events_trader ON copy_trade_events(trader_wallet);
CREATE INDEX idx_copy_events_market ON copy_trade_events(market_id);
CREATE INDEX idx_copy_events_detected ON copy_trade_events(detected_at);
