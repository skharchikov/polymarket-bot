use chrono::{DateTime, NaiveDate, Utc};
use sqlx::{PgPool, Postgres, QueryBuilder};

use super::{
    BOT_EXPR,
    positions::{Filter, push_filter},
};

#[derive(Debug, Default)]
pub struct HistoryFilter {
    pub base: Filter,
    pub outcome: Option<String>,
    pub from: Option<NaiveDate>,
    pub to: Option<NaiveDate>,
}

fn push_history(qb: &mut QueryBuilder<Postgres>, mode: &str, f: &HistoryFilter) {
    qb.push(" FROM bets WHERE resolved AND mode = ")
        .push_bind(mode.to_string());
    push_filter(qb, &f.base);
    match f.outcome.as_deref() {
        Some("won") => {
            qb.push(" AND won");
        }
        Some("lost") => {
            qb.push(" AND NOT won");
        }
        _ => {}
    }
    if let Some(d) = f.from {
        qb.push(" AND resolved_at >= ")
            .push_bind(d.and_hms_opt(0, 0, 0).unwrap().and_utc());
    }
    if let Some(d) = f.to {
        qb.push(" AND resolved_at < ").push_bind(
            d.succ_opt()
                .unwrap_or(d)
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc(),
        );
    }
}

#[derive(Debug, sqlx::FromRow)]
pub struct ResolvedRow {
    pub id: i32,
    pub placed_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub bot: String,
    pub source: String,
    pub strategy: String,
    pub category: Option<String>,
    pub question: String,
    pub side: String,
    pub entry_price: f64,
    pub cost: f64,
    pub fee_paid: f64,
    pub won: Option<bool>,
    pub pnl: Option<f64>,
    pub url: String,
}

pub async fn resolved(
    db: &PgPool,
    mode: &str,
    f: &HistoryFilter,
    limit: i64,
    offset: i64,
) -> sqlx::Result<(Vec<ResolvedRow>, i64)> {
    let mut qb = QueryBuilder::new(format!(
        "SELECT id, placed_at, resolved_at, {BOT_EXPR} AS bot, source, strategy, category, question, side, \
         entry_price, cost, fee_paid, won, pnl, url"
    ));
    push_history(&mut qb, mode, f);
    qb.push(" ORDER BY resolved_at DESC NULLS LAST, id DESC LIMIT ")
        .push_bind(limit)
        .push(" OFFSET ")
        .push_bind(offset);
    let rows = qb.build_query_as().fetch_all(db).await?;

    let mut cq = QueryBuilder::new("SELECT COUNT(*)");
    push_history(&mut cq, mode, f);
    let total: i64 = cq.build_query_scalar().fetch_one(db).await?;
    Ok((rows, total))
}

#[derive(Debug, sqlx::FromRow)]
pub struct BreakdownRow {
    pub bot: String,
    pub source: String,
    pub strategy: String,
    pub category: String,
    pub n: i64,
    pub wins: i64,
    pub pnl: f64,
    pub roi: Option<f64>,
}

pub async fn breakdown(
    db: &PgPool,
    mode: &str,
    f: &HistoryFilter,
) -> sqlx::Result<Vec<BreakdownRow>> {
    let mut qb = QueryBuilder::new(format!(
        "SELECT {BOT_EXPR} AS bot, source, strategy, COALESCE(category, '—') AS category, COUNT(*) AS n, \
         COUNT(*) FILTER (WHERE won) AS wins, COALESCE(SUM(pnl), 0)::float8 AS pnl, \
         (SUM(pnl) / NULLIF(SUM(cost + fee_paid), 0) * 100)::float8 AS roi"
    ));
    push_history(&mut qb, mode, f);
    qb.push(" GROUP BY 1, 2, 3, 4 ORDER BY pnl ASC, bot, source, strategy, category");
    qb.build_query_as().fetch_all(db).await
}
