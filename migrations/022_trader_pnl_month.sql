-- Store the followed trader's last-month Polymarket PnL alongside all-time pnl,
-- for the /traders display (leaderboard timePeriod=MONTH).
ALTER TABLE followed_traders ADD COLUMN IF NOT EXISTS pnl_month DOUBLE PRECISION;
