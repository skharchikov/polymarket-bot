# XGBoost Integration Plan

Replace the LLM-based probability estimation with a trained XGBoost model.
The LLM consensus (3 agents, ~$0.10/market, 63s rate-limited) becomes instant and free.

## Why

- LLM probability estimates are noisy — wide error bars on numerical outputs
- Polymarket prices adjust within minutes; RSS polling can't beat that with vibes
- XGBoost gives calibrated probabilities trained on actual outcomes
- Feature importance tells you what actually matters (maybe news freshness >> sentiment)
- Trains in seconds, infers in microseconds, no GPU needed

## Phase 1: Historical Data Scraper

New subcommand: `cargo run -- scrape`

Pulls resolved Polymarket history via Gamma API.

### New table: `historical_markets`

```sql
CREATE TABLE historical_markets (
    market_id       TEXT PRIMARY KEY,
    question        TEXT NOT NULL,
    category        TEXT,
    outcome_yes     BOOLEAN NOT NULL,
    volume          DOUBLE PRECISION,
    liquidity       DOUBLE PRECISION,
    end_date        TIMESTAMPTZ,
    created_at      TIMESTAMPTZ,
    resolved_at     TIMESTAMPTZ,
    price_history   JSONB  -- array of {t, p} ticks
);
```

**Target:** ~5,000 resolved markets over 6-12 months.
Rate-limit requests to avoid hammering their API.

## Phase 2: Feature Snapshot Table

Stores feature vectors at point-in-time (when the bot assesses a market).

### New table: `market_snapshots`

```sql
CREATE TABLE market_snapshots (
    id              SERIAL PRIMARY KEY,
    market_id       TEXT NOT NULL,
    snapshot_at     TIMESTAMPTZ NOT NULL,

    -- Price features
    yes_price               DOUBLE PRECISION,
    price_1h_ago            DOUBLE PRECISION,
    price_6h_ago            DOUBLE PRECISION,
    price_24h_ago           DOUBLE PRECISION,
    price_momentum_1h       DOUBLE PRECISION,
    price_momentum_24h      DOUBLE PRECISION,
    price_volatility_24h    DOUBLE PRECISION,

    -- Market features
    volume                  DOUBLE PRECISION,
    volume_24h_change       DOUBLE PRECISION,
    liquidity               DOUBLE PRECISION,
    days_to_expiry          DOUBLE PRECISION,
    category                TEXT,

    -- News features (from BM25 pipeline)
    news_count              INTEGER,
    best_bm25_score         DOUBLE PRECISION,
    avg_news_age_hours      DOUBLE PRECISION,
    news_sentiment          DOUBLE PRECISION,  -- LLM-extracted, -1 to +1

    -- Label
    outcome_yes             BOOLEAN,

    UNIQUE(market_id, snapshot_at)
);
```

**Backfill:** Walk through each historical market's price history, generate snapshots every 6h.
News features are NULL for historical data — XGBoost handles missing values natively.

**Live data:** The bot creates a snapshot every time it assesses a market (before calling the LLM).

## Phase 3: Feature Engineering

New file: `src/features.rs`

```rust
pub struct MarketFeatures {
    // Price
    pub yes_price: f64,
    pub momentum_1h: f64,
    pub momentum_24h: f64,
    pub volatility_24h: f64,

    // Market
    pub volume: f64,
    pub volume_24h_change: f64,
    pub liquidity: f64,
    pub days_to_expiry: f64,
    pub is_crypto: bool,
    pub is_politics: bool,
    pub is_sports: bool,

    // News
    pub news_count: usize,
    pub best_bm25_score: f64,
    pub avg_news_age_hours: f64,
    pub news_sentiment: f64,
}

impl MarketFeatures {
    /// Fixed-order vector for XGBoost input.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.yes_price,
            self.momentum_1h,
            self.momentum_24h,
            self.volatility_24h,
            self.volume.ln(),
            self.volume_24h_change,
            self.liquidity.ln(),
            self.days_to_expiry,
            self.is_crypto as u8 as f64,
            self.is_politics as u8 as f64,
            self.is_sports as u8 as f64,
            self.news_count as f64,
            self.best_bm25_score,
            self.avg_news_age_hours,
            self.news_sentiment,
        ]
    }
}
```

Built from existing data:
- Price history ticks → momentum + volatility
- `GammaMarket` → volume, liquidity, category, expiry
- `NewsMatch` → news_count, best BM25 score, avg freshness
- Sentiment: one LLM call per news batch (cheap, not per market)

## Phase 4: Training Pipeline

New file: `scripts/train_model.py`

```python
import os
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

# 1. Pull snapshots
conn = psycopg2.connect(os.environ["DATABASE_URL"])
df = pd.read_sql(
    "SELECT * FROM market_snapshots WHERE outcome_yes IS NOT NULL",
    conn,
)

# 2. Feature matrix
feature_cols = [
    "yes_price", "price_momentum_1h", "price_momentum_24h",
    "price_volatility_24h", "volume", "volume_24h_change",
    "liquidity", "days_to_expiry", "news_count",
    "best_bm25_score", "avg_news_age_hours", "news_sentiment",
]
X = df[feature_cols].values
y = df["outcome_yes"].astype(int).values

# 3. Time-series CV (no future leakage)
tscv = TimeSeriesSplit(n_splits=5)

# 4. Train
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
)

# 5. Calibrate with isotonic regression
calibrated = CalibratedClassifierCV(model, cv=tscv, method="isotonic")
calibrated.fit(X, y)

# 6. Evaluate per fold
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    model.fit(X[train_idx], y[train_idx])
    probs = model.predict_proba(X[test_idx])[:, 1]
    prices = df.iloc[test_idx]["yes_price"].values

    brier = brier_score_loss(y[test_idx], probs)
    ll = log_loss(y[test_idx], probs)

    # The real test: is xgb_prob - market_price profitable?
    edges = probs - prices
    correct = ((probs > 0.5) == y[test_idx]).mean()

    print(f"Fold {fold}: Brier={brier:.4f} LogLoss={ll:.4f} Acc={correct:.2%}")
    print(f"  Avg edge vs market: {edges.mean():+.4f}")

# 7. Export for Rust
model.save_model("model/xgb_model.json")

# 8. Feature importance
print("\nFeature importance:")
for name, imp in sorted(
    zip(feature_cols, model.feature_importances_),
    key=lambda x: -x[1],
):
    print(f"  {name}: {imp:.4f}")
```

**Key decisions:**
- `TimeSeriesSplit` not random — prevents future leakage
- Isotonic calibration on top — raw XGBoost probs aren't calibrated for betting
- Real metric: `xgb_prob - market_price` expectation, not just accuracy

## Phase 5: Rust Inference

Load the exported JSON model. ~100KB, loads in milliseconds.

New file: `src/model.rs`

```rust
pub struct XgbModel {
    trees: Vec<Tree>,
}

impl XgbModel {
    pub fn load(path: &str) -> Result<Self> { /* parse JSON tree dump */ }

    pub fn predict_prob(&self, features: &MarketFeatures) -> f64 {
        let raw: f64 = self.trees.iter()
            .map(|t| t.predict(&features.to_vec()))
            .sum();
        sigmoid(raw)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

Options for the inference crate:
- Hand-roll tree traversal from JSON (~200 lines, zero deps)
- Use `smartcore` crate (has gradient boosting, pure Rust)
- Use `tract` for ONNX export from Python

## Phase 6: Replace LLM in Pipeline

In `src/scanner/live.rs`, the change is surgical:

```rust
// BEFORE: 3 LLM calls ($0.10+), 63 seconds (rate limits)
let estimate = self.assess_news_impact(&nm, price, &summary, past_bets).await?;

// AFTER: instant, free
let features = MarketFeatures::from_market_and_news(&nm, price, &history);
let xgb_prob = self.model.predict_prob(&features);
let confidence = self.model.confidence(&features);
let estimate = BayesianEstimate {
    prior: price,
    posterior: xgb_prob,
    combined_lr: xgb_prob / price,
    confidence,
    reasoning: format!("XGBoost: {:.1}%", xgb_prob * 100.0),
};
```

Everything downstream (edge computation, Kelly sizing, strategy filters) stays the same —
they all operate on `BayesianEstimate` already.

## Phase 7: Hybrid Transition

Run both in parallel during the transition. Log XGBoost predictions alongside LLM.

```rust
let llm_estimate = self.assess_news_impact(...).await?;
let xgb_estimate = self.model.predict(...);

// Log both for comparison
self.log_estimate_pair(market, &llm_estimate, &xgb_estimate).await?;

// Switch when XGBoost Brier score beats LLM
let estimate = if self.model.has_sufficient_data() {
    xgb_estimate
} else {
    llm_estimate
};
```

**When to switch:** XGBoost Brier score on held-out test set < LLM Brier score from `llm_estimates` table.

## Implementation Order

```
Month 1-2: Collect live data (already happening)
     │
     ▼
Phase 1: Scraper ──→ Phase 2: Snapshot table
     │                    │
     ▼                    ▼
Phase 3: Feature eng ←───┘
     │
     ▼
Phase 4: Train (Python) ──→ model/xgb_model.json
     │
     ▼
Phase 5: Rust inference
     │
     ▼
Phase 6: Hybrid mode ──→ Full switch when Brier improves
```

## Cost Comparison

| | LLM (current) | XGBoost (target) |
|---|---|---|
| Per-market cost | ~$0.10 (3 API calls) | $0.00 |
| Latency | ~63s (rate limits) | <1ms |
| Calibration | Needs 20+ resolved estimates | Needs 1,000+ snapshots |
| Adaptability | Zero-shot on new topics | Needs retraining |
| Cold start | Works immediately | Needs months of data |

The LLM stays useful for:
- Sentiment extraction from headlines (one cheap batch call)
- Cold-start on new market categories with no training data
- Generating human-readable reasoning for Telegram messages
