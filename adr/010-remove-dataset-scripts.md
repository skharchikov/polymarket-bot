# ADR 010 — Remove Jon-Becker Dataset Training Scripts

**Status**: active
**Date**: 2026-04-09
**Supersedes**: parts of ADR 009 Phase 3

---

## Context

ADR 009 introduced `train_from_dataset.py` and `check_dataset.py` to train the ML
model on the Jon-Becker prediction-market-analysis dataset (36 GiB, 310M+ trades,
383K resolved markets). The dataset was processed on a Hetzner volume and produced
50K training samples.

After training and evaluating the model on this data, we found critical limitations:

### 1. No timestamps on trades
The Parquet trade files have a `timestamp` column but it is entirely `NULL`. Without
timestamps, we cannot compute time-series features (momentum, volatility, RSI) which
are core model inputs. All six time-series features were set to zero.

### 2. SHAP confirmed the model was just echoing market price
`yes_price` SHAP importance: **1.6166** vs **0.0731** for the next feature (`log_volume`).
The model was ~95% relying on the current price, which is equivalent to trusting the
market — no added value.

### 3. API-based training has all features
`fetch_data.py` fetches price history from the CLOB API and computes real momentum,
volatility, and RSI at each snapshot point. NLP features are derived from question
text at training time. This produces complete 29-feature training samples.

### 4. Multi-input merging was incompatible
We added `--input` multi-file support to merge the Jon-Becker dataset (zero
time-series) with API data (real time-series). This would teach the model that
time-series features are usually zero (94% of merged rows), weakening their signal.

---

## Decision

Remove Jon-Becker dataset scripts and multi-input merging. Keep all other ADR 009
improvements (NLP features, signal filters, calibration changes) which are
dataset-independent.

---

## Changes

- Deleted `scripts/train_from_dataset.py`
- Deleted `scripts/check_dataset.py`
- Reverted `train_model.py` `--input` from `nargs="+"` back to single file
- Reverted `serve_model.py` `training_data_extra.json` merge logic

---

## What We Kept From the Experiment

- **Validation of NLP features**: SHAP on the 50K dataset confirmed `q_length`,
  `q_avg_word_len`, `q_has_year`, `q_sentiment_pos` carry signal (ADR 009 §2d)
- **`is_sports` validation**: confirmed non-zero SHAP (0.0115) on a dataset with
  real sports market representation
- The dataset files on the Hetzner volume (`/mnt/HC_Volume_105339755/`) can be
  reprocessed if timestamps become available in a future dataset release
