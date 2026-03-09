-- Fix global bankroll to match sum of strategy bankrolls.
-- Previously starting_bankroll was hardcoded to 300 regardless of strategy count.
-- The init_strategy_bankrolls code now syncs this on every startup,
-- but this migration fixes the existing DB state.

-- Set global bankroll = sum of all bankroll:* keys
UPDATE portfolio SET value_f64 = (
    SELECT COALESCE(SUM(value_f64), 300.0)
    FROM portfolio
    WHERE key LIKE 'bankroll:%'
) WHERE key = 'bankroll';

-- Set starting_bankroll = number_of_strategies * 300
UPDATE portfolio SET value_f64 = (
    SELECT COUNT(*) * 300.0
    FROM portfolio
    WHERE key LIKE 'bankroll:%'
) WHERE key = 'starting_bankroll';
