# ADR 009 — Model Profitability Overhaul: Signal Filters + Recalibration

**Status**: active
**Date**: 2026-04-06
**Branch**: main

---

## Context

After 1 month of production (386 resolved XGBoost bets, 2026-03-09 to 2026-04-06):

| Metric | Value |
|--------|-------|
| Total bets | 386 |
| Win rate | 34.5% |
| Total PnL | **-€691** |
| ROI | -13.8% |
| Brier skill vs market | **-37.8%** (model worse than market price) |
| Avg PnL/bet | -€1.79 |

Comprehensive analysis of all resolved bets identified the root causes:

### 1. Sports/esports dominate and lose money
- 77% of bets (297/386) are sports/esports (detected by "vs." pattern)
- Combined PnL: **-€631**
- The model has zero predictive edge on competitive matchups
- Academic research confirms sports markets have calibration slope 1.08 (near-efficient)

### 2. YES side is systematically unprofitable
- YES bets: 220 bets, 30% WR, **-€745**
- NO bets: 166 bets, 40.4% WR, **+€54**
- Model overestimates YES probability across all calibration buckets

### 3. Calibration is broken at every probability level
- Gap between predicted and actual: -12% to -25% in every decile bucket
- Model thinks 65% → actual 40%. Thinks 45% → actual 23%.
- Overconfidence pattern is universal, not limited to specific ranges

### 4. Profitable segment exists: Non-sports NO
- 28 bets, 46.4% WR, **+€293**, avg +€10.47/bet
- High vol + NO: 100% WR in non-sports
- RSI 0.5-0.9 + NO: 57 bets, 42.1% WR, +€499

### 5. Feature correlation analysis (non-sports)
| Feature | Correlation with Won |
|---------|---------------------|
| entry_price | r=+0.57 |
| kelly_size | r=+0.37 |
| momentum_24h | r=+0.32 |
| momentum_1h | r=+0.31 |
| volatility_24h | r=+0.22 |
| is_crypto | r=+0.16 |
| days_to_expiry | r=-0.16 |

### 6. External research findings
Analysis of open-source Polymarket bots (NavnoorBawa, ReMax paper, oracle3) revealed:
- Calibration gating (only bet when edge > calibration error) captures 95% of profit
- Sports markets are near-efficiently priced (no alpha)
- Politics markets are structurally underconfident (slope 1.31 = alpha opportunity)
- Price-derived features may circularly encode market price (leakage concern)

---

## Changes

### Phase 1: Signal Filters (Rust)

#### 1a. Block sports/esports markets (configurable)
- Added `GammaMarket::is_sports_or_esports()` in `common/src/data/models.rs`
- Detects "vs.", "vs ", "spread:", "o/u", "over/under", "win on 2" patterns
- Applied in market eligibility filter in `scanner/live.rs`
- Controlled by `BLOCK_SPORTS` env var (default: true)

#### 1b. Block YES side for XGBoost signals (configurable)
- Added gate after side selection in `scanner/live.rs`
- Only blocks XGBoost source (LLM signals unaffected)
- Controlled by `BLOCK_YES_SIDE` env var (default: true)
- Rejected signals logged to `rejected_signals` table for regret analysis

#### 1c. Raised Kelly minimum from 0.003 to 0.02
- Previous: kelly_size <= 0.003 → reject
- New: kelly_size <= MIN_KELLY_SIZE (default 0.02)
- Eliminates 122 low-conviction noise bets (27.9% WR, -€289)

#### 1d. Minimum bet price filter
- New gate: bet_price < MIN_BET_PRICE (default 0.15) → reject
- Longshot entries < 15c had < 10% win rate

### Phase 2: Model Recalibration

#### 2a. Bayesian LR damping
- Changed: `dampen_lr(lr, ml_conf)` → `dampen_lr(lr, ml_conf * lr_damping)`
- `LR_DAMPING` env var (default: 0.5)
- Halves the effective confidence, anchoring posterior closer to market price
- Addresses -37.8% Brier skill by reducing model authority

#### 2b. Terminal risk scaling flipped
- Previous: penalized short-dated bets (sqrt(dte/14), full size at 14d+)
- New: penalizes long-dated bets (full size at ≤3d, sqrt(3/dte) for longer)
- Data: >7d bets had 19% WR (-€7.33/bet avg), 1-3d was the sweet spot

#### 2c. Re-added `is_sports` feature (v4, 12 → 13 features)
- Was removed in v3 for zero SHAP importance
- Root cause: training data contained no sports markets
- Production: 77% of bets are sports — model needs this signal
- Added to `MarketFeatures`, `FEATURE_COLS`, `CATEGORY_COLS`, `FeatureMap`
- Target-encoded by sidecar (binary → historical YES rate)

#### 2d. Added 16 NLP features (v5, 13 → 29 features)
- Inspired by NavnoorBawa (88-92% accuracy on high-confidence predictions using 54 non-price features)
- Extracts question text features with no price leakage:
  - `q_length`, `q_word_count`, `q_avg_word_len`, `q_word_diversity`
  - `q_has_number`, `q_has_year`, `q_has_percent`, `q_has_dollar`, `q_has_date`
  - `q_starts_will`, `q_has_by`, `q_has_before`, `q_has_above`
  - `q_sentiment_pos`, `q_sentiment_neg`, `q_certainty`
- Added `NlpFeatures` struct + `extract_nlp_features()` in `common/src/model/features.rs`
- Computed at training time in `train_model.py` from question text
- Computed at inference time in Rust and sent to sidecar

### Phase 3: Data & Infrastructure

#### 3a. Category column on bets
- Migration 020: `ALTER TABLE bets ADD COLUMN category TEXT`
- Populated from Gamma API on bet insertion
- Eliminates need for regex heuristics in future analysis

#### 3b. Reference repos cloned to `tmp/`
- NavnoorBawa/polymarket-prediction-system (54 non-price features, double calibration)
- humanplane/cross-market-state-fusion (PPO RL on Polymarket)
- YichengYang-Ethan/oracle3 (Wang Transform pricing)
- Jon-Becker/prediction-market-analysis (36 GiB trade dataset)
- Polymarket/agents (official agent framework)
- sstklen/trump-code (brute-force signal discovery)

---

## Config Env Vars Added

| Var | Default | Purpose |
|-----|---------|---------|
| `BLOCK_SPORTS` | true | Block sports/esports markets from ML betting |
| `BLOCK_YES_SIDE` | true | Block YES-side bets from XGBoost signals |
| `LR_DAMPING` | 0.5 | Bayesian LR damping multiplier (0-1, lower = trust market more) |
| `MIN_KELLY_SIZE` | 0.02 | Minimum Kelly fraction to emit signal |
| `MIN_BET_PRICE` | 0.15 | Minimum bet price (entry side) |

---

## Simulated Results (on historical 386 bets)

| Filter | N | Win Rate | PnL |
|--------|---|----------|-----|
| Baseline (all xgboost) | 386 | 34.5% | -€691 |
| NO + no sports + entry≥0.2 + kelly≥2% | 23 | 56.5% | **+€347** |
| RSI 0.5-0.9 + NO | 57 | 42.1% | **+€499** |
| NO + no sports + kelly≥2% | 25 | 52.0% | +€317 |

---

## Files Modified

- `common/src/data/models.rs` — added `is_sports_or_esports()`
- `common/src/model/features.rs` — re-added `is_sports` feature (12 → 13)
- `common/src/storage/portfolio.rs` — added `category` field to `NewBet`
- `common/src/storage/postgres.rs` — store category on bet insertion
- `trading-bot/src/config.rs` — 5 new env vars
- `trading-bot/src/scanner/live.rs` — sports filter, YES-block, Kelly gate, min price, LR damping
- `trading-bot/src/strategy.rs` — flipped terminal risk scaling
- `trading-bot/src/cycles/bet_scan.rs` — added category to NewBet
- `trading-bot/src/cycles/alerts.rs` — added category to NewBet
- `copy-trading-bot/src/cycles/copy_trade.rs` — added category to NewBet
- `scripts/train_model.py` — added is_sports feature + target encoding
- `scripts/serve_model.py` — added is_sports to FeatureMap + FEATURE_NAMES
- `scripts/fetch_data.py` — store question in snapshots for sports detection
- `migrations/020_bet_category.sql` — category column

---

## Next Steps

1. **Retrain model** with `is_sports` feature using production bet data
2. **Deploy with reduced bankroll** (€50/strategy) for 1-week validation
3. **Investigate non-price features** (per NavnoorBawa: question NLP, semantic flags)
4. **Implement calibration gating** (per ReMax: only bet when edge > calibration error)
5. **Consider double calibration** (Platt scaling + isotonic regression)
6. **Explore Wang Transform pricing** (per oracle3) as alternative to Bayesian updating
