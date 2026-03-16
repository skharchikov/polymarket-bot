# ADR 006 — Remove Dead Features (15 → 12)

**Status**: active
**Date**: 2026-03-16
**Branch**: model/remove-dead-features

---

## Context

ADR 005 introduced SHAP analysis which revealed three features with zero mean |SHAP|:

| Feature | SHAP (v2 model) | Description |
|---------|-----------------|-------------|
| `log_liquidity` | 0.000 | Log of market liquidity |
| `is_politics` | 0.000 | Binary politics category flag |
| `is_sports` | 0.000 | Binary sports category flag |

Zero SHAP means the model assigns these features no predictive weight whatsoever.
Keeping them wastes capacity, adds noise, and increases the risk of overfitting.

---

## Change

Removed `log_liquidity`, `is_politics`, `is_sports` from the feature vector: **15 → 12 features**.

- `MarketFeatures` struct (Rust) — 3 fields removed
- `MarketFeatures::NAMES` and `to_vec()` updated
- `train_model.py` `FEATURE_COLS` and `CATEGORY_COLS` updated
- `serve_model.py` `FEATURE_NAMES` and `FeatureMap` updated
- Model retrained on same v3 dataset (1500 markets / 3338 snapshots)

`is_crypto` retained — SHAP 0.005, non-zero signal.

---

## Metrics

| Metric | Before (15 features) | After (12 features) |
|--------|----------------------|---------------------|
| CV Avg Brier | 0.1573 | 0.1582 (≈ flat) |
| CV Avg PnL / fold | €+16,591 | €+16,673 (↑ 0.5%) |
| Held-out Brier | 0.0892 | **0.0879** (↓ 1.5%) |
| Held-out Accuracy | 91.3% | 90.7% (≈ flat) |

## SHAP (12-feature model)

| Rank | Feature | SHAP |
|------|---------|------|
| 1 | yes_price | 0.610 |
| 2 | log_volume | 0.454 |
| 3 | created_to_expiry_span | 0.365 |
| 4 | rsi | 0.170 |
| 5 | volatility_24h | 0.118 |
| 6 | momentum_24h | 0.100 |
| 7 | days_to_expiry | 0.071 |
| 8 | days_since_created | 0.068 |
| 9 | price_change_1d | 0.057 |
| 10 | price_change_1w | 0.007 |
| 11 | momentum_1h | 0.023 |
| 12 | is_crypto | 0.005 |

---

## Conclusion

Removing zero-importance features had no negative effect — held-out Brier improved
marginally (0.0892 → 0.0879). The model is leaner, faster to train, and has fewer
dimensions to overfit on. `is_crypto` has low but non-zero SHAP and is retained
for now; it may be a removal candidate in a future ADR if it remains near-zero.
