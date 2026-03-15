# ADR 001 — Baseline Model Audit

**Status**: active
**Date**: 2026-03-15
**Branch**: main (pre-fixes)

---

## Context

After deploying the model to production and accumulating 69 resolved bets,
win rate was 31.9% (22/69) with total PnL of -€175. This prompted a full
audit of the training pipeline vs live inference to find root causes.

---

## Live Bet Results (production, as of 2026-03-15)

| Metric | Value |
|--------|-------|
| Total resolved bets | 69 |
| Wins | 22 (31.9%) |
| Losses | 47 |
| Total PnL | -€175.71 |
| Strategies | xgboost/aggressive (46 bets), xgboost/balanced (23 bets) |

**YES vs NO breakdown:**
| Side | Wins | Losses | Win rate | Avg entry |
|------|------|--------|----------|-----------|
| YES | 4 | 26 | 13.3% | 33.5¢ |
| NO | 18 | 21 | 46.2% | 57.7¢ YES price |

**Calibration (estimated_prob vs actual win rate):**
| Prob bucket | Count | Actual win rate |
|-------------|-------|-----------------|
| 0.2–0.5 | 16 | 0% |
| 0.6 | 14 | 28.6% |
| 0.7 | 25 | 44.0% |
| 0.8 | 10 | 40.0% |
| 0.9 | 4 | 75.0% |

Model is systematically overconfident by ~25–30pp across the board.

---

## Offline CV Results (training_data.json, 1674 snapshots, 500 markets, generated 2026-03-09)

```
Feature matrix: 1674 samples x 18 features
Class balance: 47.0% YES / 53.0% NO

Time-series cross-validation:
  Fold 0: Brier=0.2011 LogLoss=0.5950 Acc=72.0% F1=0.70 Edge=+0.083 PnL=€+5178 (267 bets)
  Fold 1: Brier=0.1019 LogLoss=0.3565 Acc=88.5% F1=0.86 Edge=+0.046 PnL=€+9201 (267 bets)
  Fold 2: Brier=0.1085 LogLoss=0.3701 Acc=87.1% F1=0.84 Edge=+0.035 PnL=€+10516 (270 bets)
  Fold 3: Brier=0.2269 LogLoss=0.6909 Acc=69.2% F1=0.72 Edge=+0.121 PnL=€+6711 (272 bets)
  Fold 4: Brier=0.2522 LogLoss=0.7093 Acc=61.6% F1=0.61 Edge=-0.054 PnL=€+1994 (276 bets)

Avg Brier:    0.1781
Avg Edge:    +0.0461
Avg PnL/fold: €+6720 (simulated, flat €300 bankroll)
Total CV bets: 1352

Held-out test (last 20%, 335 samples):
  Brier:    0.1637
  Accuracy: 78.8%
  New-market subset (223 samples):
    Brier:          0.2072
    Market baseline: 0.2470  ← model beats market baseline on unseen markets
    Accuracy:        70.0%

Calibration curve (predicted → actual):
  0.10 → 0.00
  0.33 → 0.02
  0.51 → 0.47
  0.70 → 0.80
  0.88 → 1.00
```

**Feature importance (XGBoost):**
```
yes_price       0.297  ← dominant
log_volume      0.169
days_to_expiry  0.129
volatility_24h  0.122
momentum_24h    0.103
rsi             0.098
momentum_1h     0.081
log_liquidity   0.000  ← unused
is_crypto       0.000  ← unused
is_politics     0.000  ← unused
is_sports       0.000  ← unused
news_count      0.000  ← unused (always 0 in training)
best_news_score 0.000  ← unused (always 0 in training)
avg_news_age_hours 0.000
order_imbalance 0.000  ← unused (always 0 in training)
spread          0.000  ← unused (always 0 in training)
price_change_1d 0.000
price_change_1w 0.000
```

---

## Key Findings

**The offline metrics look good (70% accuracy on unseen markets, beats market baseline Brier).
Live results are terrible. This gap reveals training/live inconsistencies.**

1. **News + order book features always 0 in training, non-zero in live** — the scaler
   was fit on zero-distributions for 3 features; live non-zero values land out of distribution.
   XGBoost assigns zero importance to these features anyway, so they're dead weight.

2. **days_to_expiry mismatch** — training includes snapshots from full market lifecycle
   (median 4.3d, but range 0–304d). Live only bets at 3–14d remaining. The model
   learned patterns across all stages; cheap-YES recovery patterns from day 30+ don't
   apply at day 7.

3. **Category features unused** — `is_crypto/politics/sports` all zero importance
   despite keyword mismatch between training and live. Moot point since XGBoost
   ignores them, but indicates the category encoding is never learning.

4. **Volume is static in training** — total resolved volume repeated across all
   snapshots of the same market. Not representative of current volume at bet time.

5. **Dominant feature is `yes_price` (30%)** — the model essentially learned
   "high price = YES, low price = NO", which is tautological for resolved markets
   near resolution. At 3–14d out this is not predictive.

---

## Decision

Proceed with ADR 002: retrain addressing the top structural issues.
Do not deploy changes to main until after/after metrics confirm improvement.
