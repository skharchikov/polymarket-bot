use admin::db::signals::{
    copy_events, copy_skip_counts, correlation_blocked, near_misses, reasons,
};
use sqlx::PgPool;

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn signals_queries(pool: PgPool) {
    sqlx::query(
        "INSERT INTO rejected_signals (market_id, question, reason, edge, created_at) VALUES \
        ('a','A','edge_below_floor',0.019,NOW()), ('b','B','edge_below_floor',0.005,NOW()), \
        ('c','C','sports',NULL,NOW() - INTERVAL '3 days')",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO correlation_blocked (market_id, question, side, reason) VALUES ('d','D','NO','same event')")
        .execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO copy_trade_events (trader_wallet, market_id, condition_id, side, price, size_usd, acted_on, skip_reason) VALUES \
        ('0x1','m','c','YES',0.4,10,true,NULL), ('0x1','m2','c2','NO',0.6,10,false,'price_drift')").execute(&pool).await.unwrap();

    let r = reasons(&pool).await.unwrap();
    let edge = r.iter().find(|x| x.reason == "edge_below_floor").unwrap();
    assert_eq!((edge.n_24h, edge.n_7d), (2, 2));
    let sports = r.iter().find(|x| x.reason == "sports").unwrap();
    assert_eq!((sports.n_24h, sports.n_7d), (0, 1));

    let nm = near_misses(&pool, 10).await.unwrap();
    assert_eq!(nm[0].market_id, "a", "highest edge first");
    assert_eq!(correlation_blocked(&pool, 10).await.unwrap().len(), 1);
    assert_eq!(copy_events(&pool, 10).await.unwrap().len(), 2);
    assert_eq!(
        copy_skip_counts(&pool).await.unwrap()[0].reason,
        "price_drift"
    );
}
