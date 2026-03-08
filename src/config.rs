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
    /// Max signals (bets) per day.
    #[config(env = "MAX_SIGNALS_PER_DAY", default = 3)]
    pub max_signals_per_day: usize,

    /// Slippage assumption as a fraction (0.01 = 1%).
    #[config(env = "SLIPPAGE_PCT", default = 0.01)]
    pub slippage_pct: f64,

    /// Fee assumption as a fraction (0.02 = 2%).
    #[config(env = "FEE_PCT", default = 0.02)]
    pub fee_pct: f64,

    /// Minimum bet size in EUR.
    #[config(env = "MIN_BET", default = 10.0)]
    pub min_bet: f64,

    // --- Scanner filters ---
    /// Minimum market volume to consider.
    #[config(env = "MIN_VOLUME", default = 5000.0)]
    pub min_volume: f64,

    /// Minimum order book depth (USD) to pass liquidity filter.
    #[config(env = "MIN_BOOK_DEPTH", default = 200.0)]
    pub min_book_depth: f64,

    /// Kelly criterion fraction (0.25 = quarter-Kelly).
    #[config(env = "KELLY_FRACTION", default = 0.25)]
    pub kelly_fraction: f64,

    /// Max days until market expiry to consider.
    #[config(env = "MAX_DAYS_TO_EXPIRY", default = 14)]
    pub max_days_to_expiry: i64,

    /// Max markets to send to LLM per scan cycle.
    #[config(env = "MAX_LLM_CANDIDATES", default = 3)]
    pub max_llm_candidates: usize,

    /// Minimum effective edge (edge * confidence) to emit a signal.
    #[config(env = "MIN_EFFECTIVE_EDGE", default = 0.08)]
    pub min_effective_edge: f64,

    /// LLM model to use for news impact assessment.
    #[config(env = "LLM_MODEL", default = "gpt-4o")]
    pub llm_model: String,
}

impl AppConfig {
    pub fn load() -> Result<Self, confique::Error> {
        Self::builder().env().load()
    }
}
