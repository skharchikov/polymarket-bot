-- Track all model predictions for accuracy measurement (Brier score).
-- Both XGBoost and LLM predictions go here for unified tracking.
CREATE TABLE IF NOT EXISTS prediction_log (
    id SERIAL PRIMARY KEY,
    market_id TEXT NOT NULL,
    source TEXT NOT NULL,           -- 'xgboost', 'llm_consensus'
    market_price DOUBLE PRECISION NOT NULL,
    model_prob DOUBLE PRECISION NOT NULL,
    posterior DOUBLE PRECISION NOT NULL,  -- after Bayesian anchoring
    confidence DOUBLE PRECISION NOT NULL,
    edge DOUBLE PRECISION NOT NULL,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    outcome BOOLEAN,                -- TRUE = YES won
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_prediction_log_market ON prediction_log(market_id);
CREATE INDEX IF NOT EXISTS idx_prediction_log_resolved ON prediction_log(resolved);
