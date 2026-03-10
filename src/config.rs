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

    // --- Early exit ---
    /// Stop-loss: exit if unrealized loss exceeds this fraction of cost.
    /// Set to 1.0 to disable (let all bets run to resolution for Brier data).
    #[config(env = "STOP_LOSS_PCT", default = 1.0)]
    pub stop_loss_pct: f64,

    /// Exit if position is underwater and fewer than this many days to expiry.
    /// Set to 0 to disable expiry exits.
    #[config(env = "EXIT_DAYS_BEFORE_EXPIRY", default = 0)]
    pub exit_days_before_expiry: i64,

    /// ML model sidecar URL (Python ensemble server).
    /// When set, uses the full stacking ensemble instead of local XGBoost.
    #[config(env = "MODEL_SIDECAR_URL", default = "")]
    pub model_sidecar_url: String,
}

impl AppConfig {
    pub fn load() -> Result<Self, confique::Error> {
        Self::builder().env().load()
    }
}
