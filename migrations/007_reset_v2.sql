-- Full reset: clear all bets, bankrolls, daily snapshots, and signal counters.
-- Fresh start with 18-feature model and terminal risk scaling.

TRUNCATE bets RESTART IDENTITY CASCADE;
TRUNCATE daily_snapshots RESTART IDENTITY CASCADE;

UPDATE portfolio SET value_f64 = 300 WHERE key = 'bankroll';
UPDATE portfolio SET value_f64 = 300 WHERE key LIKE 'bankroll:%';
UPDATE portfolio SET value_f64 = 0 WHERE key LIKE 'signals_sent_today%';
UPDATE portfolio SET value_text = '' WHERE key = 'last_signal_date';
UPDATE portfolio SET value_text = '' WHERE key = 'last_daily_report_date';
