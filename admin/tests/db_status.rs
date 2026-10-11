use admin::db::status::{activity_times, attention_counts};
use sqlx::PgPool;

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn empty_db_has_no_activity_and_zero_counts(pool: PgPool) {
    let a = activity_times(&pool).await.unwrap();
    assert!(a.trading_last.is_none());
    assert!(a.copy_last.is_none());
    let c = attention_counts(&pool, "paper").await.unwrap();
    assert_eq!(c.copy_skips_6h, 0);
    assert_eq!(c.ml_bets_24h, 0);
    assert!(c.open_end_dates.is_empty());
}

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn counts_rejections_and_copy_activity(pool: PgPool) {
    sqlx::query("INSERT INTO rejected_signals (market_id, question, reason) VALUES ('m','q','edge_below_floor'),('m2','q','edge_below_floor'),('m3','q','sports')")
        .execute(&pool).await.unwrap();
    sqlx::query(
        "INSERT INTO followed_traders (proxy_wallet, last_checked_at) VALUES ('0xabc', NOW())",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO copy_trade_events (trader_wallet, market_id, condition_id, side, price, size_usd, skip_reason) VALUES ('0xabc','m','c','YES',0.4,10,'price_drift')")
        .execute(&pool).await.unwrap();

    let a = activity_times(&pool).await.unwrap();
    assert!(a.trading_last.is_some());
    assert!(a.copy_last.is_some());
    let c = attention_counts(&pool, "paper").await.unwrap();
    assert_eq!(c.ml_rejections_24h, 3);
    assert_eq!(c.top_reject_reason.as_deref(), Some("edge_below_floor"));
    assert_eq!(c.copy_skips_6h, 1);
}
