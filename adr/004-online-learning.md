# ADR 004 — Online Learning: Feature Store + Warm-Start Retraining

**Status**: draft
**Date**: 2026-03-15
**Branch**: tbd

---

## Context

The current training pipeline has a fundamental data quality problem for our own resolved bets:
`fetch_data.py` re-fetches price history for each resolved bet and reconstructs the feature
vector at bet-placement time by interpolation. This reconstruction is an approximation —
the exact market state (volume, liquidity, RSI, momentum) at the moment we placed the bet
is not preserved.

Additionally, the model retrains daily from scratch on 3000 Polymarket historical markets.
Our own resolved bets (highest-quality samples — we have exact entry context and known outcome)
are diluted in a bulk retrain and reconstructed with noise.

Two goals:
1. **Exact feature capture**: store the precise 16-feature vector at bet-placement time.
2. **Warm-start retraining**: when new bets resolve, update the model incrementally without
   a full cold retrain — keeping the model responsive to live performance between daily retrains.

---

## Decision

### Feature Store

Add a `bet_features` table storing the exact feature vector as JSONB at the moment a bet
is placed:

```sql
CREATE TABLE bet_features (
    bet_id      INTEGER PRIMARY KEY REFERENCES bets(id) ON DELETE CASCADE,
    features    JSONB NOT NULL,   -- {"yes_price": 0.65, "momentum_1h": 0.02, ...}
    version     INTEGER NOT NULL DEFAULT 1,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

JSONB chosen over `DOUBLE PRECISION[]` for extensibility:
- Adding/removing features does not break old records — missing keys default to `0.0`
- Self-describing: `fetch_data.py` reads features by name, not array index
- `version` field allows filtering records by feature schema generation when feature set changes

### Retraining Schedule

```
Every resolved bet     →  recalibration check (future, lightweight)
Every 10 resolved bets →  warm-start retrain on stored bet_features only (seconds, no API)
Daily (nightly cron)   →  full cold retrain: fetch 3000 markets + all stored bets
```

The warm-start retrain:
1. Queries `bet_features JOIN bets WHERE resolved=true` — exact features + known outcomes
2. Loads existing `model/xgb_model.json` via `xgb_model=` parameter — adds trees on top
3. Refits calibration layer on the same data
4. Hot-reloads model in sidecar — no container restart

Daily cold retrain remains the ground truth. Warm-starts correct for drift between retrains
without compounding errors over many cycles.

### `fetch_data.py` update

When `bet_features` records exist for resolved bets, use them directly instead of
reconstructing from price history. Falls back to reconstruction for bets placed before
this ADR (no stored features yet).

---

## Metrics Before

No CV/held-out change expected at training time — this is a data pipeline change.
The base model sees the same historical market data it always has.

**Current own-bet data quality:**
- 69 resolved bets total (as of ADR 001)
- Features reconstructed from price history approximation
- Reconstruction error: unknown, but volume/liquidity at bet time vs. at training time
  may differ significantly (Gamma API stores current values, not historical)

**Why no before/after CV numbers here:**
The benefit is not visible in offline CV — it accrues over time as resolved bets with
exact features accumulate. Each warm-start retrain will use progressively cleaner data.
Calibration improvement will be visible in live bet performance (win rate, Brier score
on actual outcomes) over weeks, not in a single eval run.

---

## Metrics After

To be filled in after 4–6 weeks of production feature collection.

Target signals:
- Live calibration: estimated_prob bucket accuracy (currently ~25–30pp overconfident, per ADR 001)
- Own-bet Brier score improvement between cold retrain and warm-start-adjusted model
- Warm-start retrain latency (target: <30s end-to-end)

---

## Trade-offs Considered

**Why not full retrain on every N bets?**
Full retrain = fetch 3000 markets from Polymarket API + train 5-model ensemble with
calibration. Takes 10–20 minutes. Doing this on every 10 resolved bets (which could be
every few days) is wasteful and risks rate-limiting the Polymarket API. Daily scheduled
retrain is sufficient as the ground truth; warm-start handles intra-day drift.

**Why not true online learning (e.g. `river` library)?**
XGBoost/LGB ensemble doesn't support single-sample updates natively. `river` would require
replacing the entire model architecture. At our scale (~5-10 resolved bets/week), true
online learning offers no meaningful advantage over warm-start retraining on accumulated
resolved bets.

**JSONB storage cost:**
Each bet_features row is ~500 bytes. At 500 bets/year this is negligible.

---

## Conclusion

Pending implementation and 4–6 weeks of data collection.
