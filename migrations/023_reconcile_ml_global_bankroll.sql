-- Reconcile the ML bot's global bankroll after fixing the copy-bet global-credit
-- drift (copy resolutions were crediting the global `bankroll` though copy entries
-- never debited it, inflating global from ~€1.2k to ~€25k).
--
-- Model: the ML bot and copy bot are FULLY SEPARATE. The global `bankroll` and
-- `starting_bankroll` track ONLY the ML strategies (aggressive/balanced/conservative).
-- Copy strategies live entirely in their own `bankroll:copy:*` / `starting_bankroll:copy:*`
-- keys and never touch the global keys.
--
-- All statements are guarded with EXISTS so this is a no-op on a fresh DB (where the
-- per-strategy keys don't exist yet and init_strategy_bankrolls will seed them).

-- 1. Global current bankroll = sum of ML per-strategy bankrolls (undo copy inflation).
UPDATE portfolio SET value_f64 = (
    SELECT COALESCE(SUM(value_f64), 0)
    FROM portfolio
    WHERE key IN ('bankroll:aggressive', 'bankroll:balanced', 'bankroll:conservative')
)
WHERE key = 'bankroll'
  AND EXISTS (SELECT 1 FROM portfolio WHERE key = 'bankroll:balanced');

-- 2. Per-strategy ML starting capital = 300 seed + 1000 (migration 017 top-up) = 1300.
--    Migration 017 only touched bankroll:* and left these starting keys at 300.
UPDATE portfolio SET value_f64 = 1300.0
WHERE key IN ('starting_bankroll:aggressive',
              'starting_bankroll:balanced',
              'starting_bankroll:conservative');

-- 3. Global starting_bankroll = sum of ML per-strategy starting (= 3 * 1300 = 3900).
UPDATE portfolio SET value_f64 = (
    SELECT COALESCE(SUM(value_f64), 0)
    FROM portfolio
    WHERE key IN ('starting_bankroll:aggressive',
                  'starting_bankroll:balanced',
                  'starting_bankroll:conservative')
)
WHERE key = 'starting_bankroll'
  AND EXISTS (SELECT 1 FROM portfolio WHERE key = 'starting_bankroll:balanced');
