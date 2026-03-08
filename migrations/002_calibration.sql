CREATE TABLE IF NOT EXISTS llm_estimates (
    id                      SERIAL PRIMARY KEY,
    market_id               TEXT NOT NULL,
    question                TEXT NOT NULL,
    agent_role              TEXT NOT NULL,
    raw_probability         DOUBLE PRECISION NOT NULL,
    raw_confidence          DOUBLE PRECISION NOT NULL,
    consensus_probability   DOUBLE PRECISION,
    consensus_confidence    DOUBLE PRECISION,
    current_price           DOUBLE PRECISION NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved                BOOLEAN NOT NULL DEFAULT FALSE,
    outcome                 BOOLEAN,
    resolved_at             TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_llm_estimates_market ON llm_estimates(market_id);
CREATE INDEX IF NOT EXISTS idx_llm_estimates_resolved ON llm_estimates(resolved);
