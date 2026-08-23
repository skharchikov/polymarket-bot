-- Make Telegram subscribers per-bot so each bot only messages (and prunes) its own users.
ALTER TABLE telegram_users ADD COLUMN IF NOT EXISTS bot        TEXT NOT NULL DEFAULT 'trading';
ALTER TABLE telegram_users ADD COLUMN IF NOT EXISTS active     BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE telegram_users ADD COLUMN IF NOT EXISTS blocked_at TIMESTAMPTZ;

-- Repoint the primary key from (chat_id) to (bot, chat_id).
ALTER TABLE telegram_users DROP CONSTRAINT IF EXISTS telegram_users_pkey;
ALTER TABLE telegram_users ADD PRIMARY KEY (bot, chat_id);

-- Backfill: existing rows default to bot='trading'. Duplicate each into 'copy'
-- so no current subscriber silently loses messages from either bot.
INSERT INTO telegram_users (chat_id, username, first_name, last_seen, created_at, bot, active)
SELECT chat_id, username, first_name, last_seen, created_at, 'copy', active
FROM telegram_users
WHERE bot = 'trading'
ON CONFLICT (bot, chat_id) DO NOTHING;
