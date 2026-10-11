use chrono::{DateTime, Utc};
use sqlx::PgPool;

#[derive(Debug, sqlx::FromRow)]
pub struct ReasonCount {
    pub reason: String,
    pub n_24h: i64,
    pub n_7d: i64,
}

pub async fn reasons(db: &PgPool) -> sqlx::Result<Vec<ReasonCount>> {
    sqlx::query_as(
        "SELECT reason, COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') AS n_24h, COUNT(*) AS n_7d \
         FROM rejected_signals WHERE created_at > NOW() - INTERVAL '7 days' GROUP BY reason ORDER BY n_7d DESC",
    )
    .fetch_all(db)
    .await
}

#[derive(Debug, sqlx::FromRow)]
pub struct NearMiss {
    pub created_at: DateTime<Utc>,
    pub market_id: String,
    pub question: String,
    pub reason: String,
    pub current_price: Option<f64>,
    pub estimated_prob: Option<f64>,
    pub edge: Option<f64>,
    pub confidence: Option<f64>,
}

/// Highest-edge rejections in the last 24h: what almost got bet.
pub async fn near_misses(db: &PgPool, limit: i64) -> sqlx::Result<Vec<NearMiss>> {
    sqlx::query_as(
        "SELECT created_at, market_id, question, reason, current_price, estimated_prob, edge, confidence \
         FROM rejected_signals WHERE created_at > NOW() - INTERVAL '24 hours' AND edge IS NOT NULL \
         ORDER BY edge DESC LIMIT $1",
    )
    .bind(limit)
    .fetch_all(db)
    .await
}

#[derive(Debug, sqlx::FromRow)]
pub struct Blocked {
    pub created_at: DateTime<Utc>,
    pub question: String,
    pub side: String,
    pub reason: String,
}

pub async fn correlation_blocked(db: &PgPool, limit: i64) -> sqlx::Result<Vec<Blocked>> {
    sqlx::query_as(
        "SELECT created_at, question, side, reason FROM correlation_blocked ORDER BY created_at DESC LIMIT $1",
    )
    .bind(limit)
    .fetch_all(db)
    .await
}

#[derive(Debug, sqlx::FromRow)]
pub struct CopyEvent {
    pub detected_at: DateTime<Utc>,
    pub trader_wallet: String,
    pub username: Option<String>,
    pub market_id: String,
    pub side: String,
    pub price: f64,
    pub size_usd: f64,
    pub acted_on: bool,
    pub skip_reason: Option<String>,
}

pub async fn copy_events(db: &PgPool, limit: i64) -> sqlx::Result<Vec<CopyEvent>> {
    sqlx::query_as(
        "SELECT e.detected_at, e.trader_wallet, t.username, e.market_id, e.side, e.price, e.size_usd, e.acted_on, e.skip_reason \
         FROM copy_trade_events e LEFT JOIN followed_traders t ON LOWER(t.proxy_wallet) = LOWER(e.trader_wallet) \
         ORDER BY e.detected_at DESC LIMIT $1",
    )
    .bind(limit)
    .fetch_all(db)
    .await
}

#[derive(Debug, sqlx::FromRow)]
pub struct SkipCount {
    pub reason: String,
    pub n: i64,
}

pub async fn copy_skip_counts(db: &PgPool) -> sqlx::Result<Vec<SkipCount>> {
    sqlx::query_as(
        "SELECT skip_reason AS reason, COUNT(*) AS n FROM copy_trade_events \
         WHERE skip_reason IS NOT NULL AND detected_at > NOW() - INTERVAL '24 hours' GROUP BY 1 ORDER BY n DESC",
    )
    .fetch_all(db)
    .await
}
