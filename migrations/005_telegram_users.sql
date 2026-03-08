CREATE TABLE IF NOT EXISTS telegram_users (
    chat_id     TEXT PRIMARY KEY,
    username    TEXT,
    first_name  TEXT,
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
