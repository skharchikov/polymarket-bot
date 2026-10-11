use std::collections::BTreeMap;

use chrono::{DateTime, NaiveDate, Utc};
use serde::Serialize;
use sqlx::PgPool;

use super::BOT_EXPR;

#[derive(Debug, sqlx::FromRow)]
pub struct Kpis {
    pub bankroll_trading: f64,
    pub bankroll_copy: f64,
    pub open_exposure: f64,
    pub open_count: i64,
    pub pnl_24h: f64,
    pub resolved_24h: i64,
    pub pnl_7d: f64,
    pub resolved_7d: i64,
    pub pnl_all: f64,
    pub roi_all_pct: Option<f64>,
    pub win_rate_30d: Option<f64>,
    pub resolved_30d: i64,
}

pub async fn kpis(db: &PgPool, mode: &str) -> sqlx::Result<Kpis> {
    sqlx::query_as(
        "SELECT \
           (SELECT COALESCE(SUM(value_f64), 0) FROM portfolio WHERE key LIKE 'bankroll:%' AND key NOT LIKE 'bankroll:copy:%') AS bankroll_trading, \
           (SELECT COALESCE(SUM(value_f64), 0) FROM portfolio WHERE key LIKE 'bankroll:copy:%') AS bankroll_copy, \
           COALESCE(SUM(cost + fee_paid) FILTER (WHERE NOT resolved), 0) AS open_exposure, \
           COUNT(*) FILTER (WHERE NOT resolved) AS open_count, \
           COALESCE(SUM(pnl) FILTER (WHERE resolved AND resolved_at > NOW() - INTERVAL '24 hours'), 0) AS pnl_24h, \
           COUNT(*) FILTER (WHERE resolved AND resolved_at > NOW() - INTERVAL '24 hours') AS resolved_24h, \
           COALESCE(SUM(pnl) FILTER (WHERE resolved AND resolved_at > NOW() - INTERVAL '7 days'), 0) AS pnl_7d, \
           COUNT(*) FILTER (WHERE resolved AND resolved_at > NOW() - INTERVAL '7 days') AS resolved_7d, \
           COALESCE(SUM(pnl) FILTER (WHERE resolved), 0) AS pnl_all, \
           SUM(pnl) FILTER (WHERE resolved) / NULLIF(SUM(cost + fee_paid) FILTER (WHERE resolved), 0) * 100 AS roi_all_pct, \
           AVG(CASE WHEN won THEN 100.0::float8 ELSE 0.0::float8 END) \
             FILTER (WHERE resolved AND resolved_at > NOW() - INTERVAL '30 days') AS win_rate_30d, \
           COUNT(*) FILTER (WHERE resolved AND resolved_at > NOW() - INTERVAL '30 days') AS resolved_30d \
         FROM bets WHERE mode = $1",
    )
    .bind(mode)
    .fetch_one(db)
    .await
}

#[derive(Debug, sqlx::FromRow)]
pub struct EquityRow {
    pub day: NaiveDate,
    pub bot: String,
    pub pnl: f64,
}

pub async fn equity_rows(db: &PgPool, mode: &str) -> sqlx::Result<Vec<EquityRow>> {
    sqlx::query_as(sqlx::AssertSqlSafe(format!(
        "SELECT (resolved_at AT TIME ZONE 'UTC')::date AS day, {BOT_EXPR} AS bot, SUM(pnl)::float8 AS pnl \
         FROM bets WHERE mode = $1 AND resolved AND pnl IS NOT NULL AND resolved_at > NOW() - INTERVAL '90 days' \
         GROUP BY 1, 2 ORDER BY 1"
    )))
    .bind(mode)
    .fetch_all(db)
    .await
}

#[derive(Debug, Serialize)]
pub struct EquitySeries {
    /// Unix seconds (uPlot x-axis).
    pub days: Vec<i64>,
    pub trading: Vec<f64>,
    pub copy: Vec<f64>,
}

/// Turn per-day PnL rows into aligned cumulative series per bot.
pub fn cumulate(rows: &[EquityRow]) -> EquitySeries {
    let mut by_day: BTreeMap<NaiveDate, (f64, f64)> = BTreeMap::new();
    for r in rows {
        let e = by_day.entry(r.day).or_default();
        if r.bot == "copy" {
            e.1 += r.pnl
        } else {
            e.0 += r.pnl
        }
    }
    let (mut t, mut c) = (0.0, 0.0);
    let mut s = EquitySeries {
        days: vec![],
        trading: vec![],
        copy: vec![],
    };
    for (day, (dt, dc)) in by_day {
        t += dt;
        c += dc;
        s.days
            .push(day.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp());
        s.trading.push(t);
        s.copy.push(c);
    }
    s
}

#[derive(Debug, sqlx::FromRow)]
pub struct ActivityRow {
    pub ts: DateTime<Utc>,
    pub bot: String,
    pub label: String,
    pub side: String,
    pub price: f64,
    pub amount: f64,
    pub kind: String,
    pub pnl: Option<f64>,
    pub detail: Option<String>,
}

pub async fn activity(db: &PgPool, mode: &str) -> sqlx::Result<Vec<ActivityRow>> {
    sqlx::query_as(sqlx::AssertSqlSafe(format!(
        "SELECT * FROM ( \
           (SELECT placed_at AS ts, {BOT_EXPR} AS bot, question AS label, side, entry_price AS price, cost AS amount, \
                   'placed' AS kind, NULL::float8 AS pnl, NULL::text AS detail \
            FROM bets WHERE mode = $1 ORDER BY placed_at DESC LIMIT 20) \
           UNION ALL \
           (SELECT resolved_at, {BOT_EXPR}, question, side, entry_price, cost, \
                   CASE WHEN won THEN 'won' ELSE 'lost' END, pnl, NULL \
            FROM bets WHERE mode = $1 AND resolved AND resolved_at IS NOT NULL ORDER BY resolved_at DESC LIMIT 20) \
           UNION ALL \
           (SELECT detected_at, 'copy', market_id, side, price, size_usd, 'skipped', NULL, skip_reason \
            FROM copy_trade_events WHERE skip_reason IS NOT NULL ORDER BY detected_at DESC LIMIT 20) \
         ) a ORDER BY ts DESC LIMIT 20"
    )))
    .bind(mode)
    .fetch_all(db)
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn cumulate_builds_aligned_running_totals() {
        let rows = vec![
            EquityRow {
                day: d("2026-10-01"),
                bot: "trading".into(),
                pnl: 10.0,
            },
            EquityRow {
                day: d("2026-10-01"),
                bot: "copy".into(),
                pnl: -5.0,
            },
            EquityRow {
                day: d("2026-10-03"),
                bot: "copy".into(),
                pnl: 2.0,
            },
        ];
        let s = cumulate(&rows);
        assert_eq!(s.days.len(), 2);
        assert_eq!(s.trading, vec![10.0, 10.0]);
        assert_eq!(s.copy, vec![-5.0, -3.0]);
        assert_eq!(
            s.days[0],
            d("2026-10-01")
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp()
        );
    }

    #[test]
    fn cumulate_empty() {
        let s = cumulate(&[]);
        assert!(s.days.is_empty() && s.trading.is_empty() && s.copy.is_empty());
    }
}
