-- Reset all bets and bankrolls to fresh state (€300 per strategy).
-- This migration is idempotent via the guard check.

DO $$
BEGIN
  -- Only run if there are bets to clear (guard against re-runs)
  IF EXISTS (SELECT 1 FROM bets) THEN
    DELETE FROM bets;
    UPDATE portfolio SET value = '300' WHERE key = 'bankroll';
    UPDATE portfolio SET value = '300' WHERE key LIKE 'bankroll:%';
    UPDATE portfolio SET value = '0' WHERE key LIKE 'signals_sent_today%';
    RAISE NOTICE 'Reset: cleared bets, bankrolls set to €300';
  END IF;
END $$;
