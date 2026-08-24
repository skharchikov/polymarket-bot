use confique::Config;

#[derive(Debug, Config)]
pub struct CopyTradingConfig {
    /// Postgres connection string.
    #[config(env = "DATABASE_URL")]
    pub database_url: String,

    // --- Telegram ---
    #[config(env = "TELEGRAM_BOT_TOKEN")]
    pub telegram_bot_token: String,

    #[config(env = "TELEGRAM_CHAT_ID")]
    pub telegram_chat_id: String,

    // --- Copy trading ---
    /// Copy trading poll interval in minutes.
    #[config(env = "COPY_TRADE_INTERVAL_MINS", default = 1)]
    pub copy_trade_interval_mins: u64,

    // --- Betting ---
    /// Slippage assumption as a fraction (0.01 = 1%).
    #[config(env = "SLIPPAGE_PCT", default = 0.01)]
    pub slippage_pct: f64,

    /// Fee assumption as a fraction (0.02 = 2%).
    #[config(env = "FEE_PCT", default = 0.02)]
    pub fee_pct: f64,

    /// Port for the Prometheus metrics HTTP endpoint.
    #[config(env = "METRICS_PORT", default = 9001)]
    pub metrics_port: u16,

    // --- Daily top-traders digest (advisory) ---
    /// Whether to broadcast the daily top-traders digest.
    #[config(env = "DIGEST_ENABLED", default = true)]
    pub digest_enabled: bool,

    /// How many traders to include in the daily digest.
    #[config(env = "DIGEST_TOP_N", default = 5)]
    pub digest_top_n: usize,

    /// UTC hour (0-23) at which the daily digest is sent.
    #[config(env = "DIGEST_HOUR_UTC", default = 9)]
    pub digest_hour_utc: u32,
}

impl CopyTradingConfig {
    pub fn load() -> Result<Self, confique::Error> {
        Self::builder().env().load()
    }
}
