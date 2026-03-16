use confique::Config;

#[derive(Debug, Config)]
pub struct AppConfig {
    /// Postgres connection string.
    #[config(env = "DATABASE_URL")]
    pub database_url: String,

    // --- Telegram ---
    #[config(env = "TELEGRAM_BOT_TOKEN")]
    pub telegram_bot_token: String,

    #[config(env = "TELEGRAM_CHAT_ID")]
    pub telegram_chat_id: String,

    // --- Scan intervals ---
    /// Housekeeping loop interval in minutes (resolution checks, daily reports).
    #[config(env = "SCAN_INTERVAL_MINS", default = 30)]
    pub scan_interval_mins: u64,

    /// News scan loop interval in minutes.
    #[config(env = "NEWS_SCAN_INTERVAL_MINS", default = 10)]
    pub news_scan_interval_mins: u64,

    /// Bet scan loop interval in minutes (market scoring + betting).
    #[config(env = "BET_SCAN_INTERVAL_MINS", default = 10)]
    pub bet_scan_interval_mins: u64,

    /// Enable news fetching and embedding-based matching.
    /// When the model sidecar is active, news has no effect on predictions —
    /// disable to save RSS + OpenAI embedding costs.
    #[config(env = "NEWS_ENABLED", default = false)]
    pub news_enabled: bool,

    // --- Betting ---
    /// Slippage assumption as a fraction (0.01 = 1%).
    #[config(env = "SLIPPAGE_PCT", default = 0.01)]
    pub slippage_pct: f64,

    /// Fee assumption as a fraction (0.02 = 2%).
    #[config(env = "FEE_PCT", default = 0.02)]
    pub fee_pct: f64,

    // --- Scanner filters ---
    /// Minimum market volume to consider.
    #[config(env = "MIN_VOLUME", default = 1000.0)]
    pub min_volume: f64,

    /// Minimum order book depth (USD) to pass liquidity filter.
    #[config(env = "MIN_BOOK_DEPTH", default = 200.0)]
    pub min_book_depth: f64,

    /// Kelly criterion fraction (0.25 = quarter-Kelly).
    #[config(env = "KELLY_FRACTION", default = 0.25)]
    pub kelly_fraction: f64,

    /// Max days until market expiry to consider.
    /// Sweet spot is 3-14d: enough signal, still uncertain, good training data.
    #[config(env = "MAX_DAYS_TO_EXPIRY", default = 14)]
    pub max_days_to_expiry: i64,

    /// Max markets to send to LLM per scan cycle (each costs consensus_agents API calls).
    #[config(env = "MAX_LLM_CANDIDATES", default = 1)]
    pub max_llm_candidates: usize,

    /// Top N markets from XGBoost ranking to consider for betting.
    #[config(env = "MAX_MODEL_CANDIDATES", default = 15)]
    pub max_model_candidates: usize,

    /// Minimum effective edge (edge * confidence) to emit a signal.
    #[config(env = "MIN_EFFECTIVE_EDGE", default = 0.08)]
    pub min_effective_edge: f64,

    /// LLM model to use for news impact assessment.
    #[config(env = "LLM_MODEL", default = "gpt-4o")]
    pub llm_model: String,

    /// Heartbeat interval in minutes (0 to disable).
    #[config(env = "HEARTBEAT_INTERVAL_MINS", default = 60)]
    pub heartbeat_interval_mins: u64,

    /// Expected model retrain interval in hours (matches RETRAIN_MAX_AGE_HOURS).
    #[config(env = "RETRAIN_INTERVAL_HOURS", default = 24)]
    pub retrain_interval_hours: u64,

    // --- Multi-agent consensus ---
    /// Number of LLM agents for consensus (1=single, 2-3=multi-agent).
    #[config(env = "CONSENSUS_AGENTS", default = 2)]
    pub consensus_agents: usize,

    /// Min resolved estimates before applying calibration correction.
    #[config(env = "CALIBRATION_MIN_SAMPLES", default = 20)]
    pub calibration_min_samples: usize,

    // --- Market fetch ---
    /// Max markets to fetch from Polymarket API per scan.
    #[config(env = "MAX_MARKETS_FETCH", default = 1000)]
    pub max_markets_fetch: usize,

    /// Minimum YES price to consider (filters out near-certain NO).
    #[config(env = "MIN_PRICE", default = 0.03)]
    pub min_price: f64,

    /// Maximum YES price to consider (filters out near-certain YES).
    #[config(env = "MAX_PRICE", default = 0.97)]
    pub max_price: f64,

    /// Starting bankroll per strategy in EUR.
    #[config(env = "STRATEGY_BANKROLL", default = 300.0)]
    pub strategy_bankroll: f64,

    /// Active strategies (comma-separated: aggressive,balanced,conservative).
    #[config(env = "STRATEGIES", default = "aggressive,balanced,conservative")]
    pub strategies: String,

    // --- Strategy: aggressive ---
    #[config(env = "AGGRESSIVE_KELLY_FRACTION", default = 0.50)]
    pub aggressive_kelly_fraction: f64,
    #[config(env = "AGGRESSIVE_MIN_EDGE", default = 0.05)]
    pub aggressive_min_edge: f64,
    #[config(env = "AGGRESSIVE_MIN_CONFIDENCE", default = 0.40)]
    pub aggressive_min_confidence: f64,
    #[config(env = "AGGRESSIVE_MAX_SIGNALS", default = 10)]
    pub aggressive_max_signals: usize,
    #[config(env = "AGGRESSIVE_MIN_BET", default = 5.0)]
    pub aggressive_min_bet: f64,

    // --- Strategy: balanced ---
    #[config(env = "BALANCED_KELLY_FRACTION", default = 0.25)]
    pub balanced_kelly_fraction: f64,
    #[config(env = "BALANCED_MIN_EDGE", default = 0.06)]
    pub balanced_min_edge: f64,
    #[config(env = "BALANCED_MIN_CONFIDENCE", default = 0.40)]
    pub balanced_min_confidence: f64,
    #[config(env = "BALANCED_MAX_SIGNALS", default = 5)]
    pub balanced_max_signals: usize,
    #[config(env = "BALANCED_MIN_BET", default = 5.0)]
    pub balanced_min_bet: f64,

    // --- Strategy: conservative ---
    #[config(env = "CONSERVATIVE_KELLY_FRACTION", default = 0.15)]
    pub conservative_kelly_fraction: f64,
    #[config(env = "CONSERVATIVE_MIN_EDGE", default = 0.08)]
    pub conservative_min_edge: f64,
    #[config(env = "CONSERVATIVE_MIN_CONFIDENCE", default = 0.50)]
    pub conservative_min_confidence: f64,
    #[config(env = "CONSERVATIVE_MAX_SIGNALS", default = 3)]
    pub conservative_max_signals: usize,
    #[config(env = "CONSERVATIVE_MIN_BET", default = 15.0)]
    pub conservative_min_bet: f64,

    // --- Early exit (disabled by default — let all bets resolve for learning) ---
    /// Stop-loss: exit if unrealized loss exceeds this fraction of cost.
    /// Set to 999.0 to disable. E.g. 0.5 = exit at 50% loss.
    #[config(env = "STOP_LOSS_PCT", default = 999.0)]
    pub stop_loss_pct: f64,

    /// Exit if position is underwater (≥10% loss) and fewer than this many days to expiry.
    /// Set to 0 to disable expiry exits.
    #[config(env = "EXIT_DAYS_BEFORE_EXPIRY", default = 0)]
    pub exit_days_before_expiry: i64,

    /// ML model sidecar URL (Python ensemble server).
    /// When set, uses the full stacking ensemble instead of local XGBoost.
    #[config(env = "MODEL_SIDECAR_URL", default = "")]
    pub model_sidecar_url: String,

    /// Port for the Prometheus metrics HTTP endpoint.
    #[config(env = "METRICS_PORT", default = 9000)]
    pub metrics_port: u16,

    // --- Copy trading ---
    /// Enable copy trading cycle.
    #[config(env = "COPY_TRADE_ENABLED", default = false)]
    pub copy_trade_enabled: bool,

    /// Copy trading poll interval in minutes.
    #[config(env = "COPY_TRADE_INTERVAL_MINS", default = 5)]
    pub copy_trade_interval_mins: u64,
}

impl AppConfig {
    pub fn load() -> Result<Self, confique::Error> {
        Self::builder().env().load()
    }

    #[cfg(test)]
    pub fn test_default() -> Self {
        Self {
            database_url: String::new(),
            telegram_bot_token: String::new(),
            telegram_chat_id: String::new(),
            scan_interval_mins: 30,
            news_scan_interval_mins: 10,
            bet_scan_interval_mins: 10,
            news_enabled: false,
            slippage_pct: 0.01,
            fee_pct: 0.02,
            min_volume: 1000.0,
            min_book_depth: 200.0,
            kelly_fraction: 0.25,
            max_days_to_expiry: 14,
            max_llm_candidates: 1,
            max_model_candidates: 15,
            min_effective_edge: 0.08,
            llm_model: "gpt-4o".into(),
            heartbeat_interval_mins: 60,
            retrain_interval_hours: 24,
            consensus_agents: 2,
            calibration_min_samples: 20,
            max_markets_fetch: 1000,
            min_price: 0.03,
            max_price: 0.97,
            strategy_bankroll: 300.0,
            strategies: "aggressive,balanced,conservative".into(),
            aggressive_kelly_fraction: 0.50,
            aggressive_min_edge: 0.05,
            aggressive_min_confidence: 0.40,
            aggressive_max_signals: 10,
            aggressive_min_bet: 5.0,
            balanced_kelly_fraction: 0.25,
            balanced_min_edge: 0.06,
            balanced_min_confidence: 0.40,
            balanced_max_signals: 5,
            balanced_min_bet: 5.0,
            conservative_kelly_fraction: 0.15,
            conservative_min_edge: 0.08,
            conservative_min_confidence: 0.50,
            conservative_max_signals: 3,
            conservative_min_bet: 15.0,
            stop_loss_pct: 999.0,
            exit_days_before_expiry: 0,
            model_sidecar_url: String::new(),
            metrics_port: 9000,
            copy_trade_enabled: false,
            copy_trade_interval_mins: 5,
        }
    }
}
