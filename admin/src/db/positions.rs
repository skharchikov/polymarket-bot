use chrono::{DateTime, Utc};
use serde::Deserialize;
use sqlx::{PgPool, Postgres, QueryBuilder};

use super::BOT_EXPR;

#[derive(Debug, Default, Deserialize)]
pub struct Filter {
    pub bot: Option<String>,
    pub source: Option<String>,
    pub strategy: Option<String>,
}

/// Append `AND ...` clauses for the bot/source/strategy filter.
pub fn push_filter(qb: &mut QueryBuilder<Postgres>, f: &Filter) {
    if let Some(b) = &f.bot {
        qb.push(format!(" AND {BOT_EXPR} = ")).push_bind(b.clone());
    }
    if let Some(s) = &f.source {
        qb.push(" AND source = ").push_bind(s.clone());
    }
    if let Some(s) = &f.strategy {
        qb.push(" AND strategy = ").push_bind(s.clone());
    }
}

#[derive(Debug, sqlx::FromRow)]
pub struct OpenRow {
    pub id: i32,
    pub market_id: String,
    pub question: String,
    pub side: String,
    pub entry_price: f64,
    pub shares: f64,
    pub cost: f64,
    pub fee_paid: f64,
    pub strategy: String,
    pub source: String,
    pub url: String,
    pub end_date: Option<String>,
    pub trader: Option<String>,
    pub bot: String,
    pub placed_at: DateTime<Utc>,
}

pub async fn open_positions(db: &PgPool, mode: &str, f: &Filter) -> sqlx::Result<Vec<OpenRow>> {
    let mut qb = QueryBuilder::new(format!(
        "SELECT id, market_id, question, side, entry_price, shares, cost, fee_paid, strategy, source, url, \
         end_date, copy_ref->>'trader' AS trader, {BOT_EXPR} AS bot, placed_at \
         FROM bets WHERE NOT resolved AND mode = "
    ));
    qb.push_bind(mode.to_string());
    push_filter(&mut qb, f);
    qb.push(" ORDER BY placed_at DESC");
    qb.build_query_as().fetch_all(db).await
}

#[derive(Debug, sqlx::FromRow)]
pub struct Detail {
    pub id: i32,
    pub estimated_prob: f64,
    pub entry_price: f64,
    pub edge: f64,
    pub confidence: f64,
    pub kelly_size: f64,
    pub fee_paid: f64,
    pub reasoning: String,
    pub url: String,
    pub context: Option<String>,
    pub copy_ref: Option<String>,
    pub features: Option<String>,
}

pub async fn detail(db: &PgPool, id: i32) -> sqlx::Result<Option<Detail>> {
    sqlx::query_as(
        "SELECT b.id, b.estimated_prob, b.entry_price, b.edge, b.confidence, b.kelly_size, b.fee_paid, \
                b.reasoning, b.url, jsonb_pretty(b.context) AS context, jsonb_pretty(b.copy_ref) AS copy_ref, \
                jsonb_pretty(f.features) AS features \
         FROM bets b LEFT JOIN bet_features f ON f.bet_id = b.id WHERE b.id = $1",
    )
    .bind(id)
    .fetch_optional(db)
    .await
}

/// Distinct sources and strategies for filter dropdowns.
pub async fn filter_options(db: &PgPool, mode: &str) -> sqlx::Result<(Vec<String>, Vec<String>)> {
    let sources = sqlx::query_scalar("SELECT DISTINCT source FROM bets WHERE mode = $1 ORDER BY 1")
        .bind(mode)
        .fetch_all(db)
        .await?;
    let strategies =
        sqlx::query_scalar("SELECT DISTINCT strategy FROM bets WHERE mode = $1 ORDER BY 1")
            .bind(mode)
            .fetch_all(db)
            .await?;
    Ok((sources, strategies))
}
