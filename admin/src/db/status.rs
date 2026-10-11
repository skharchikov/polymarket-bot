use chrono::{DateTime, Utc};
use sqlx::PgPool;

pub struct ActivityTimes {
    pub trading_last: Option<DateTime<Utc>>,
    pub copy_last: Option<DateTime<Utc>>,
}

/// Last sign of life per bot. Trading writes rejections every scan; copy
/// stamps followed_traders.last_checked_at every poll.
pub async fn activity_times(db: &PgPool) -> sqlx::Result<ActivityTimes> {
    let (trading_last, copy_last): (Option<DateTime<Utc>>, Option<DateTime<Utc>>) = sqlx::query_as(
        "SELECT GREATEST( \
                 (SELECT MAX(created_at) FROM rejected_signals), \
                 (SELECT MAX(created_at) FROM prediction_log WHERE source <> 'copy_trade'), \
                 (SELECT MAX(placed_at) FROM bets WHERE strategy NOT LIKE 'copy:%')), \
               (SELECT MAX(last_checked_at) FROM followed_traders WHERE active)",
    )
    .fetch_one(db)
    .await?;
    Ok(ActivityTimes {
        trading_last,
        copy_last,
    })
}

pub struct AttentionCounts {
    pub open_end_dates: Vec<Option<String>>,
    pub copy_skips_6h: i64,
    pub ml_bets_24h: i64,
    pub ml_rejections_24h: i64,
    pub top_reject_reason: Option<String>,
}

pub async fn attention_counts(db: &PgPool, mode: &str) -> sqlx::Result<AttentionCounts> {
    let open_end_dates: Vec<Option<String>> =
        sqlx::query_scalar("SELECT end_date FROM bets WHERE NOT resolved AND mode = $1")
            .bind(mode)
            .fetch_all(db)
            .await?;
    let (copy_skips_6h, ml_bets_24h, ml_rejections_24h): (i64, i64, i64) = sqlx::query_as(
        "SELECT \
           (SELECT COUNT(*) FROM copy_trade_events WHERE skip_reason IS NOT NULL AND detected_at > NOW() - INTERVAL '6 hours'), \
           (SELECT COUNT(*) FROM bets WHERE mode = $1 AND strategy NOT LIKE 'copy:%' AND placed_at > NOW() - INTERVAL '24 hours'), \
           (SELECT COUNT(*) FROM rejected_signals WHERE created_at > NOW() - INTERVAL '24 hours')",
    )
    .bind(mode)
    .fetch_one(db)
    .await?;
    let top_reject_reason: Option<String> = sqlx::query_scalar(
        "SELECT reason FROM rejected_signals WHERE created_at > NOW() - INTERVAL '24 hours' \
         GROUP BY reason ORDER BY COUNT(*) DESC LIMIT 1",
    )
    .fetch_optional(db)
    .await?;
    Ok(AttentionCounts {
        open_end_dates,
        copy_skips_6h,
        ml_bets_24h,
        ml_rejections_24h,
        top_reject_reason,
    })
}
