-- Top up each strategy bankroll by €1000 and sync global totals.

UPDATE portfolio
SET value_f64 = value_f64 + 1000.0
WHERE key LIKE 'bankroll:%';

-- Sync global bankroll to sum of strategy bankrolls
UPDATE portfolio SET value_f64 = (
    SELECT COALESCE(SUM(value_f64), 0.0)
    FROM portfolio
    WHERE key LIKE 'bankroll:%'
) WHERE key = 'bankroll';

-- Sync starting_bankroll (add same total top-up)
UPDATE portfolio SET value_f64 = (
    SELECT COUNT(*) * 1000.0
    FROM portfolio
    WHERE key LIKE 'bankroll:%'
) + value_f64
WHERE key = 'starting_bankroll';
