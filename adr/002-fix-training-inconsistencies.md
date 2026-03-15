# ADR 002 — Fix Training/Live Pipeline Inconsistencies

**Status**: draft
**Date**: 2026-03-15
**Branch**: model/training-fixes

---

## Context

See ADR 001. The model shows good offline CV metrics but poor live performance.
Root cause analysis identified structural gaps between training data generation
and live inference. This ADR tracks the fixes and their measured impact.

---

## Changes

### 1. Drop zero-inflated features from training and inference
**Problem**: `news_count`, `best_news_score`, `avg_news_age_hours`, `order_imbalance`,
`spread` are always 0 in training. XGBoost assigns them zero importance. In live
inference they receive real values, creating distribution shift. The scaler learned
a zero-distribution for these columns and real values are outliers.

**Fix**: Remove all 5 features from `FEATURE_COLS` in `train_model.py` and from
`MarketFeatures` in `features.rs`. The live scanner will stop populating them.
This reduces the feature vector from 18 → 13 features.

**Why not keep news features with real historical data**: Polymarket's news data is
not available retroactively for historical markets. Generating synthetic non-zero
values would introduce more noise than signal.

### 2. Align category keyword detection
**Problem**: Training detects crypto via 4 keywords in `category` field only.
Live detects via 15+ keywords across `category` AND `question` text.

**Fix**: Update `train_model.py` to use the same keyword set and apply to both
`category` and `question` fields. Align `is_politics` to also check question text.

### 3. Use volume24hr instead of total volume
**Problem**: Training stores final total resolved volume (static across all snapshots
of a market). Live uses current `volume_num`. A market with $50k final volume
looks the same at day 1 ($2k current) and day 13 ($49k current).

**Fix**: Store `volume24hr` in snapshots (already in Gamma API response) as a
second volume feature, or replace `volume` with it in `fetch_data.py`.

### 4. Filter snapshots to deployment window (days_to_expiry ≤ 14)
**Problem**: Training includes snapshots at 0–304 days to expiry. Live only bets
at 3–14 days remaining. The model learned patterns from the full lifecycle;
the cheap-YES recovery at day 30 doesn't apply at day 7.

**Fix**: In `_extract_snapshot`, skip snapshots where `days_to_expiry > 14`.
This reduces training set size but makes it representative of deployment conditions.

---

## Metrics Before (from ADR 001)

| Metric | Value |
|--------|-------|
| CV Avg Brier | 0.1781 |
| CV Avg Edge | +0.046 |
| CV Avg PnL/fold | €+6720 |
| Held-out Brier | 0.1637 |
| Held-out Accuracy | 78.8% |
| New-market Brier | 0.2072 (market baseline: 0.2470) |
| New-market Accuracy | 70.0% |
| **Live win rate** | **31.9% (22/69)** |
| **Live PnL** | **-€175.71** |

---

## Metrics After

Retrained on the same 1674 snapshots with 13 features (5 dropped).
Note: `days_to_expiry` filter requires a fresh data fetch — will apply in ADR 003.
The category keyword fix doesn't change the current dataset (category field is empty
for all 1674 snapshots — needs fresh fetch with updated code to take effect).

```
Feature matrix: 1674 samples x 13 features
Class balance: 47.0% YES / 53.0% NO

Time-series cross-validation:
  Fold 0: Brier=0.2030 LogLoss=0.6023 Acc=71.3% F1=0.69 Edge=+0.083 PnL=€+5161 (276 bets)
  Fold 1: Brier=0.1028 LogLoss=0.3583 Acc=88.2% F1=0.86 Edge=+0.046 PnL=€+9294 (270 bets)
  Fold 2: Brier=0.1084 LogLoss=0.3711 Acc=87.5% F1=0.85 Edge=+0.036 PnL=€+10517 (267 bets)
  Fold 3: Brier=0.2259 LogLoss=0.6843 Acc=69.2% F1=0.72 Edge=+0.116 PnL=€+6704 (276 bets)
  Fold 4: Brier=0.2535 LogLoss=0.7124 Acc=61.6% F1=0.61 Edge=-0.057 PnL=€+1919 (273 bets)

Avg Brier:    0.1787  (was 0.1781, +0.001 — neutral)
Avg Edge:    +0.0449  (was +0.0461, -0.001 — neutral)
Avg PnL/fold: €+6719  (was €+6720 — identical)

Held-out test (335 samples):
  Brier:    0.1611  (was 0.1637 — improved)
  Accuracy: 81.5%   (was 78.8% — improved +2.7pp)
  New-market subset:
    Brier:          0.2041  (was 0.2072 — improved)
    Market baseline: 0.2470
    Accuracy:        73.5%  (was 70.0% — improved +3.5pp)

Calibration (predicted → actual):
  0.10 → 0.00  (was 0.10 → 0.00)
  0.32 → 0.06  (was 0.33 → 0.02 — slightly better)
  0.49 → 0.45  (was 0.51 → 0.47)
  0.70 → 0.78  (was 0.70 → 0.80)
  0.86 → 1.00  (was 0.88 → 1.00)
```

**Feature importance (XGBoost) after fix:**
```
yes_price       0.296  (unchanged — still dominant concern)
log_volume      0.160
volatility_24h  0.134
days_to_expiry  0.124
momentum_24h    0.109
rsi             0.107
momentum_1h     0.071
log_liquidity   0.000  ← still unused
is_crypto       0.000  ← still unused (no category data in current dataset)
is_politics     0.000
is_sports       0.000
price_change_1d 0.000
price_change_1w 0.000
```

---

## Conclusion

Dropping 5 zero-inflated features gives small but consistent improvements on held-out
accuracy (+2.7pp) and new-market Brier. CV metrics are neutral — expected, since those
snapshots have zero values for the dropped features anyway.

The real gains will come from ADR 003: fetching fresh data with `days_to_expiry ≤ 14`
filter and populated category fields. The current training set has no category data
(all empty strings) so the keyword fix has no effect yet.

**`yes_price` remains the dominant feature (30%)** — this is the core issue. The model
is essentially a calibrated market price with momentum corrections. Until we restrict
training to the 3–14d deployment window with a fresh fetch, this won't change.

**Status**: partially applied. Full effect pending ADR 003 (fresh data fetch).
