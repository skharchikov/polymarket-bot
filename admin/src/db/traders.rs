use chrono::{DateTime, Utc};
use sqlx::PgPool;

#[derive(Debug, sqlx::FromRow)]
pub struct TraderRow {
    pub proxy_wallet: String,
    pub username: Option<String>,
    pub rank: Option<i32>,
    pub pnl: Option<f64>,
    pub pnl_month: Option<f64>,
    pub active: bool,
    pub last_checked_at: Option<DateTime<Utc>>,
    pub our_bets: i64,
    pub our_open: i64,
    pub our_pnl: f64,
}

pub async fn traders(db: &PgPool, mode: &str) -> sqlx::Result<Vec<TraderRow>> {
    sqlx::query_as(
        "SELECT ft.proxy_wallet, ft.username, ft.rank, ft.pnl, ft.pnl_month, ft.active, ft.last_checked_at, \
                COALESCE(b.n, 0) AS our_bets, COALESCE(b.open_n, 0) AS our_open, COALESCE(b.pnl, 0)::float8 AS our_pnl \
         FROM followed_traders ft \
         LEFT JOIN ( \
            SELECT LOWER(copy_ref->>'wallet') AS wallet, COUNT(*) AS n, \
                   COUNT(*) FILTER (WHERE NOT resolved) AS open_n, SUM(pnl) FILTER (WHERE resolved) AS pnl \
            FROM bets WHERE copy_ref IS NOT NULL AND mode = $1 GROUP BY 1 \
         ) b ON b.wallet = LOWER(ft.proxy_wallet) \
         ORDER BY ft.active DESC, our_pnl DESC",
    )
    .bind(mode)
    .fetch_all(db)
    .await
}
