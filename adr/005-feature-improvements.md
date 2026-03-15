# ADR 005 — Feature Improvements: Temporal Features, Target Encoding, SHAP Analysis

**Status**: active
**Date**: 2026-03-15
**Branch**: model/feature-improvements

---

## Context

After the baseline audit (ADR 001–003) we had a 13-feature model with known weaknesses:

- `is_crypto`, `is_politics`, `is_sports` were binary flags with near-zero contribution.
  A binary flag carries no information about *how much* each category correlates with YES outcomes.
- No temporal context: the model had no sense of how old a market is or how long it was planned to run.
- No post-hoc explainability: we had no way to rank feature importance or detect dead features.

---

## Changes

### 1. Target encoding for category features

`is_crypto / is_politics / is_sports` (binary 0/1) replaced by their **smoothed historical YES rate**
(Bayesian smoothing with global prior, `m=10`).

- Computed during training, saved to `model/category_encoding.json`
- Applied in the Python sidecar at inference time (`_apply_encoding()`)
- Rust sends the same binary values — no Rust API change needed

### 2. Two new temporal features (13 → 15 features)

| Feature | Description |
|---------|-------------|
| `days_since_created` | Market age at snapshot time (how long it has been live) |
| `created_to_expiry_span` | Total planned duration of the market (created → end_date) |

Required adding `created_at` to `GammaMarket` and computing both fields in
`MarketFeatures::from_market_and_history()`.

### 3. SHAP analysis (post-training)

`TreeExplainer` on the XGBoost submodel. Mean |SHAP| values saved to `model/shap_summary.json`
after every training run. Used to detect dead features and validate new ones.

### 4. Typed feature API

`Signal.features` and `NewBet.features` changed from `Option<serde_json::Value>` to
`Option<MarketFeatures>` — serialised to JSONB only at the DB insert layer (`postgres.rs`).

---

## Metrics

### Before (v2 data: 1000 markets / 2963 snapshots, temporal features defaulted to 30.0)

| Metric | Value |
|--------|-------|
| CV Avg Brier | 0.1846 |
| CV Avg Accuracy | ~72% |
| CV Avg PnL / fold | €+7,889 |
| Held-out Brier | 0.1696 |
| Held-out Accuracy | 74.4% |

### After (v3 data: 1500 markets / 3338 snapshots, real temporal features)

| Metric | Value |
|--------|-------|
| CV Avg Brier | 0.1573 (↓ 14.8%) |
| CV Avg Accuracy | ~77.8% |
| CV Avg PnL / fold | €+16,591 (↑ 110%) |
| Held-out Brier | 0.0892 (↓ 47.3%) |
| Held-out Accuracy | 91.3% |
| New-market Brier | 0.1066 vs baseline 0.2255 |

### SHAP feature importance (mean |SHAP|, ranked)

| Rank | Feature | SHAP |
|------|---------|------|
| 1 | yes_price | 0.643 |
| 2 | log_volume | 0.455 |
| 3 | **created_to_expiry_span** | 0.349 |
| 4 | rsi | 0.172 |
| 5 | momentum_24h | 0.106 |
| 6 | volatility_24h | 0.102 |
| 7 | days_to_expiry | 0.083 |
| 8 | **days_since_created** | 0.081 |
| 9 | price_change_1d | 0.058 |
| 10 | momentum_1h | 0.027 |
| 11 | price_change_1w | 0.004 |
| 12 | is_crypto | 0.009 |
| 13 | log_liquidity | 0.000 |
| 14 | is_politics | 0.000 |
| 15 | is_sports | 0.000 |

---

## Conclusion

Both temporal features contribute meaningfully: `created_to_expiry_span` is the **3rd most
important feature overall**, suggesting market duration is a strong predictor of outcome
(longer-planned markets behave differently from short-lived ones).

Three features have zero SHAP importance: `log_liquidity`, `is_politics`, `is_sports`.
These are candidates for removal in a future ADR to reduce dimensionality.

The held-out accuracy jump (74% → 91%) is partly attributable to more and higher-quality
training data (1500 vs 1000 markets) in addition to the feature changes.
