-- ============================================================
-- POLYMARKET BOT — ONE-MONTH PERFORMANCE ANALYSIS
-- Run: psql $DATABASE_URL -f analysis/performance_queries.sql
-- ============================================================

-- ============================================================
-- 1. HIGH-LEVEL SCOREBOARD
-- ============================================================
SELECT
  COUNT(*)                                          AS total_bets,
  COUNT(*) FILTER (WHERE resolved)                  AS resolved,
  COUNT(*) FILTER (WHERE NOT resolved)              AS open,
  COUNT(*) FILTER (WHERE won)                       AS wins,
  COUNT(*) FILTER (WHERE won = false AND resolved)  AS losses,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won)
    / NULLIF(COUNT(*) FILTER (WHERE resolved), 0), 1) AS win_rate_pct,
  ROUND(SUM(pnl) FILTER (WHERE resolved)::numeric, 2)  AS total_pnl,
  ROUND(SUM(cost)::numeric, 2)                      AS total_wagered,
  ROUND(SUM(fee_paid)::numeric, 2)                  AS total_fees,
  ROUND(AVG(pnl) FILTER (WHERE resolved)::numeric, 3) AS avg_pnl_per_bet,
  ROUND((SUM(pnl) FILTER (WHERE resolved)
    / NULLIF(SUM(cost) FILTER (WHERE resolved), 0) * 100)::numeric, 2) AS roi_pct
FROM bets;

-- ============================================================
-- 2. PERFORMANCE BY SOURCE (xgboost / llm_consensus / copy_trade)
-- ============================================================
SELECT
  source,
  COUNT(*) FILTER (WHERE resolved)   AS resolved,
  COUNT(*) FILTER (WHERE won)        AS wins,
  COUNT(*) FILTER (WHERE won = false AND resolved) AS losses,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won)
    / NULLIF(COUNT(*) FILTER (WHERE resolved), 0), 1) AS win_rate,
  ROUND(SUM(pnl) FILTER (WHERE resolved)::numeric, 2) AS pnl,
  ROUND(AVG(pnl) FILTER (WHERE resolved)::numeric, 3) AS avg_pnl,
  ROUND(AVG(edge) FILTER (WHERE resolved)::numeric, 3) AS avg_edge
FROM bets
GROUP BY source
ORDER BY pnl DESC;

-- ============================================================
-- 3. PERFORMANCE BY STRATEGY
-- ============================================================
SELECT
  strategy,
  COUNT(*) FILTER (WHERE resolved)   AS resolved,
  COUNT(*) FILTER (WHERE won)        AS wins,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won)
    / NULLIF(COUNT(*) FILTER (WHERE resolved), 0), 1) AS win_rate,
  ROUND(SUM(pnl) FILTER (WHERE resolved)::numeric, 2) AS pnl,
  ROUND(AVG(edge)::numeric, 3)       AS avg_edge,
  ROUND(AVG(kelly_size)::numeric, 4) AS avg_kelly
FROM bets
GROUP BY strategy
ORDER BY pnl DESC;

-- ============================================================
-- 4. WEEKLY PERFORMANCE TREND
-- ============================================================
SELECT
  DATE_TRUNC('week', placed_at)::date AS week,
  COUNT(*)                             AS placed,
  COUNT(*) FILTER (WHERE resolved)     AS resolved,
  COUNT(*) FILTER (WHERE won)          AS wins,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won)
    / NULLIF(COUNT(*) FILTER (WHERE resolved), 0), 1) AS win_rate,
  ROUND(SUM(pnl) FILTER (WHERE resolved)::numeric, 2) AS pnl,
  ROUND(AVG(edge)::numeric, 3)         AS avg_edge
FROM bets
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 5. CALIBRATION — predicted prob vs actual outcome (decile bins)
-- ============================================================
SELECT
  bucket,
  cnt,
  ROUND(avg_predicted::numeric, 3) AS avg_predicted,
  ROUND(actual_rate::numeric, 3)   AS actual_rate,
  ROUND((actual_rate - avg_predicted)::numeric, 3) AS gap
FROM (
  SELECT
    WIDTH_BUCKET(estimated_prob, 0, 1, 10) AS bucket,
    COUNT(*)                               AS cnt,
    AVG(estimated_prob)                    AS avg_predicted,
    AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END) AS actual_rate
  FROM bets
  WHERE resolved
  GROUP BY 1
) t
ORDER BY bucket;

-- ============================================================
-- 6. EDGE-BUCKET ANALYSIS — does higher edge → more profit?
-- ============================================================
SELECT
  CASE
    WHEN edge < 0.05 THEN '< 5%'
    WHEN edge < 0.10 THEN '5-10%'
    WHEN edge < 0.15 THEN '10-15%'
    WHEN edge < 0.20 THEN '15-20%'
    ELSE '≥ 20%'
  END AS edge_bucket,
  COUNT(*)  AS n,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won)
    / NULLIF(COUNT(*), 0), 1) AS win_rate,
  ROUND(SUM(pnl)::numeric, 2)  AS pnl,
  ROUND(AVG(pnl)::numeric, 3)  AS avg_pnl
FROM bets
WHERE resolved
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 7. CONFIDENCE-BUCKET ANALYSIS
-- ============================================================
SELECT
  CASE
    WHEN confidence < 0.3  THEN 'low (<0.3)'
    WHEN confidence < 0.5  THEN 'med (0.3-0.5)'
    WHEN confidence < 0.7  THEN 'high (0.5-0.7)'
    ELSE 'very high (≥0.7)'
  END AS conf_bucket,
  COUNT(*)  AS n,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won)
    / NULLIF(COUNT(*), 0), 1) AS win_rate,
  ROUND(SUM(pnl)::numeric, 2)  AS pnl,
  ROUND(AVG(pnl)::numeric, 3)  AS avg_pnl
FROM bets
WHERE resolved
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 8. BRIER SCORE — model vs market (from prediction_log)
-- ============================================================
SELECT
  source,
  COUNT(*) AS n,
  ROUND(AVG(POWER(posterior  - outcome::int, 2))::numeric, 4) AS model_brier,
  ROUND(AVG(POWER(market_price - outcome::int, 2))::numeric, 4) AS market_brier,
  ROUND((1.0 - AVG(POWER(posterior - outcome::int, 2))
             / NULLIF(AVG(POWER(market_price - outcome::int, 2)), 0))::numeric * 100, 2)
    AS skill_pct
FROM prediction_log
WHERE resolved AND outcome IS NOT NULL
GROUP BY ROLLUP(source)
ORDER BY source NULLS LAST;

-- ============================================================
-- 9. YES vs NO SIDE PERFORMANCE
-- ============================================================
SELECT
  side,
  COUNT(*) FILTER (WHERE resolved) AS n,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won)
    / NULLIF(COUNT(*) FILTER (WHERE resolved), 0), 1) AS win_rate,
  ROUND(SUM(pnl) FILTER (WHERE resolved)::numeric, 2)  AS pnl,
  ROUND(AVG(entry_price) FILTER (WHERE resolved)::numeric, 3) AS avg_entry
FROM bets
GROUP BY side;

-- ============================================================
-- 10. CRYPTO vs NON-CRYPTO PERFORMANCE
-- ============================================================
SELECT
  CASE
    WHEN LOWER(question) ~ 'bitcoin|btc|ethereum|eth|crypto|solana|sol' THEN 'crypto'
    ELSE 'non-crypto'
  END AS category,
  COUNT(*) FILTER (WHERE resolved) AS n,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won)
    / NULLIF(COUNT(*) FILTER (WHERE resolved), 0), 1) AS win_rate,
  ROUND(SUM(pnl) FILTER (WHERE resolved)::numeric, 2) AS pnl
FROM bets
GROUP BY 1;

-- ============================================================
-- 11. BET DURATION vs OUTCOME
-- ============================================================
SELECT
  CASE
    WHEN EXTRACT(EPOCH FROM resolved_at - placed_at)/3600 < 24 THEN '< 1 day'
    WHEN EXTRACT(EPOCH FROM resolved_at - placed_at)/3600 < 72 THEN '1-3 days'
    WHEN EXTRACT(EPOCH FROM resolved_at - placed_at)/3600 < 168 THEN '3-7 days'
    ELSE '> 7 days'
  END AS duration,
  COUNT(*) AS n,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won) / COUNT(*), 1) AS win_rate,
  ROUND(SUM(pnl)::numeric, 2) AS pnl,
  ROUND(AVG(pnl)::numeric, 3) AS avg_pnl
FROM bets
WHERE resolved AND resolved_at IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 12. ENTRY PRICE RANGE vs OUTCOME
--     (extreme prices are harder to profit from)
-- ============================================================
SELECT
  CASE
    WHEN entry_price < 0.20 THEN '< 0.20'
    WHEN entry_price < 0.40 THEN '0.20-0.40'
    WHEN entry_price < 0.60 THEN '0.40-0.60'
    WHEN entry_price < 0.80 THEN '0.60-0.80'
    ELSE '≥ 0.80'
  END AS price_range,
  COUNT(*) AS n,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won) / COUNT(*), 1) AS win_rate,
  ROUND(SUM(pnl)::numeric, 2) AS pnl,
  ROUND(AVG(edge)::numeric, 3) AS avg_edge
FROM bets
WHERE resolved
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 13. SLIPPAGE ANALYSIS
-- ============================================================
SELECT
  ROUND(AVG(ABS(slipped_price - entry_price))::numeric, 4) AS avg_slippage,
  ROUND(MAX(ABS(slipped_price - entry_price))::numeric, 4) AS max_slippage,
  ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
    (ORDER BY ABS(slipped_price - entry_price))::numeric, 4) AS p95_slippage,
  ROUND(SUM(ABS(slipped_price - entry_price) * shares)::numeric, 2) AS total_slippage_cost
FROM bets
WHERE slipped_price IS NOT NULL;

-- ============================================================
-- 14. BIGGEST WINS & LOSSES (top 10 each)
-- ============================================================
(SELECT 'WIN' AS type, question, side, pnl, edge, confidence, source, placed_at::date
 FROM bets WHERE resolved ORDER BY pnl DESC LIMIT 10)
UNION ALL
(SELECT 'LOSS', question, side, pnl, edge, confidence, source, placed_at::date
 FROM bets WHERE resolved ORDER BY pnl ASC LIMIT 10)
ORDER BY type, pnl DESC;

-- ============================================================
-- 15. REGRET ANALYSIS — rejected signals that would have won
-- ============================================================
SELECT
  r.reason,
  COUNT(*) AS total_rejected,
  COUNT(*) FILTER (WHERE p.resolved AND p.outcome = true
    AND r.estimated_prob > r.current_price) AS would_have_won_yes,
  COUNT(*) FILTER (WHERE p.resolved AND p.outcome = false
    AND r.estimated_prob < r.current_price) AS would_have_won_no
FROM rejected_signals r
LEFT JOIN prediction_log p
  ON p.market_id = r.market_id
  AND p.source = 'xgboost'
GROUP BY r.reason
ORDER BY total_rejected DESC;

-- ============================================================
-- 16. KELLY SIZING EFFICIENCY — are bigger bets performing?
-- ============================================================
SELECT
  CASE
    WHEN kelly_size < 0.02 THEN 'tiny (<2%)'
    WHEN kelly_size < 0.05 THEN 'small (2-5%)'
    WHEN kelly_size < 0.10 THEN 'medium (5-10%)'
    ELSE 'large (≥10%)'
  END AS kelly_bucket,
  COUNT(*) AS n,
  ROUND(100.0 * COUNT(*) FILTER (WHERE won) / COUNT(*), 1) AS win_rate,
  ROUND(SUM(pnl)::numeric, 2) AS pnl,
  ROUND(SUM(cost)::numeric, 2) AS capital_deployed
FROM bets
WHERE resolved
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 17. FEATURE IMPORTANCE (proxy) — avg feature values for wins vs losses
-- ============================================================
SELECT
  'wins' AS group_label,
  ROUND(AVG((f.features->>'yes_price')::float)::numeric, 3) AS yes_price,
  ROUND(AVG((f.features->>'momentum_1h')::float)::numeric, 4) AS mom_1h,
  ROUND(AVG((f.features->>'momentum_24h')::float)::numeric, 4) AS mom_24h,
  ROUND(AVG((f.features->>'volatility_24h')::float)::numeric, 4) AS vol_24h,
  ROUND(AVG((f.features->>'rsi')::float)::numeric, 1) AS rsi,
  ROUND(AVG((f.features->>'log_volume')::float)::numeric, 2) AS log_vol,
  ROUND(AVG((f.features->>'days_to_expiry')::float)::numeric, 1) AS dte
FROM bet_features f
JOIN bets b ON b.id = f.bet_id
WHERE b.resolved AND b.won
UNION ALL
SELECT
  'losses',
  ROUND(AVG((f.features->>'yes_price')::float)::numeric, 3),
  ROUND(AVG((f.features->>'momentum_1h')::float)::numeric, 4),
  ROUND(AVG((f.features->>'momentum_24h')::float)::numeric, 4),
  ROUND(AVG((f.features->>'volatility_24h')::float)::numeric, 4),
  ROUND(AVG((f.features->>'rsi')::float)::numeric, 1),
  ROUND(AVG((f.features->>'log_volume')::float)::numeric, 2),
  ROUND(AVG((f.features->>'days_to_expiry')::float)::numeric, 1)
FROM bet_features f
JOIN bets b ON b.id = f.bet_id
WHERE b.resolved AND NOT b.won;

-- ============================================================
-- 18. DAILY P&L CURVE (for charting)
-- ============================================================
SELECT
  resolved_at::date AS day,
  COUNT(*) AS resolved,
  ROUND(SUM(pnl)::numeric, 2) AS daily_pnl,
  ROUND(SUM(SUM(pnl)) OVER (ORDER BY resolved_at::date)::numeric, 2) AS cumulative_pnl
FROM bets
WHERE resolved AND resolved_at IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 19. COPY-TRADE PERFORMANCE (if applicable)
-- ============================================================
SELECT
  ft.username AS trader,
  COUNT(*) FILTER (WHERE b.resolved) AS resolved,
  COUNT(*) FILTER (WHERE b.won) AS wins,
  ROUND(SUM(b.pnl) FILTER (WHERE b.resolved)::numeric, 2) AS pnl
FROM bets b
JOIN followed_traders ft
  ON b.copy_ref->>'proxy_wallet' = ft.proxy_wallet
WHERE b.source = 'copy_trade'
GROUP BY ft.username
ORDER BY pnl DESC;

-- ============================================================
-- 20. CORRELATION-BLOCKED — how many, and resolved outcomes
-- ============================================================
SELECT
  cb.reason,
  COUNT(DISTINCT cb.market_id) AS blocked,
  COUNT(DISTINCT cb.market_id) FILTER (WHERE p.resolved) AS resolved,
  ROUND(AVG(CASE WHEN p.outcome THEN 1.0 ELSE 0.0 END)::numeric, 2) AS outcome_rate
FROM correlation_blocked cb
LEFT JOIN prediction_log p ON p.market_id = cb.market_id
GROUP BY cb.reason
ORDER BY blocked DESC;
