-- Grant EUR1000 of betting capital to each ML strategy (aggressive/balanced/
-- conservative) — they were drained (aggressive ~EUR6) and could no longer size
-- bets. Bump BOTH bankroll and starting_bankroll so the injection is treated as
-- capital, not profit (keeps ROI honest), then resync the ML global keys.
--
-- Copy strategies are untouched (ML and copy are fully separate; see migration 023).
-- Guarded with EXISTS so this is a no-op on a fresh DB.

-- 1. Add EUR1000 to each ML strategy's current bankroll.
UPDATE portfolio SET value_f64 = value_f64 + 1000.0
WHERE key IN ('bankroll:aggressive', 'bankroll:balanced', 'bankroll:conservative');

-- 2. Add EUR1000 to each ML strategy's starting capital (injection, not profit).
UPDATE portfolio SET value_f64 = value_f64 + 1000.0
WHERE key IN ('starting_bankroll:aggressive',
              'starting_bankroll:balanced',
              'starting_bankroll:conservative');

-- 3. Resync ML global bankroll = sum of ML per-strategy bankrolls.
UPDATE portfolio SET value_f64 = (
    SELECT COALESCE(SUM(value_f64), 0)
    FROM portfolio
    WHERE key IN ('bankroll:aggressive', 'bankroll:balanced', 'bankroll:conservative')
)
WHERE key = 'bankroll'
  AND EXISTS (SELECT 1 FROM portfolio WHERE key = 'bankroll:balanced');

-- 4. Resync ML global starting_bankroll = sum of ML per-strategy starting.
UPDATE portfolio SET value_f64 = (
    SELECT COALESCE(SUM(value_f64), 0)
    FROM portfolio
    WHERE key IN ('starting_bankroll:aggressive',
                  'starting_bankroll:balanced',
                  'starting_bankroll:conservative')
)
WHERE key = 'starting_bankroll'
  AND EXISTS (SELECT 1 FROM portfolio WHERE key = 'starting_bankroll:balanced');
