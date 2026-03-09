-- Fix starting_bankroll to 900 (3 strategies × €300).
-- Previous bug was syncing it to current bankroll sum instead of initial.
UPDATE portfolio SET value_f64 = 900.0 WHERE key = 'starting_bankroll';
