use admin::db::history::{HistoryFilter, breakdown, resolved};
use sqlx::PgPool;

async fn seed(pool: &PgPool) {
    for (i, (strategy, won, pnl)) in [
        ("balanced", true, 10.0),
        ("balanced", false, -20.0),
        ("copy:0xab", true, 5.0),
    ]
    .into_iter()
    .enumerate()
    {
        sqlx::query(
            "INSERT INTO bets (market_id, question, entry_price, slipped_price, shares, cost, fee_paid, \
             estimated_prob, confidence, edge, kelly_size, reasoning, strategy, source, resolved, won, pnl, resolved_at) \
             VALUES ($1,'Q',0.5,0.5,10,20,0.4,0.6,0.5,0.1,0.05,'r',$2,'xgboost',true,$3,$4, NOW() - ($5 || ' days')::interval)",
        )
        .bind(format!("m{i}")).bind(strategy).bind(won).bind(pnl).bind(i.to_string())
        .execute(pool).await.unwrap();
    }
}

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn pagination_filters_and_breakdown(pool: PgPool) {
    seed(&pool).await;
    let (rows, total) = resolved(&pool, "paper", &HistoryFilter::default(), 2, 0)
        .await
        .unwrap();
    assert_eq!(total, 3);
    assert_eq!(rows.len(), 2);

    let lost = HistoryFilter {
        outcome: Some("lost".into()),
        ..Default::default()
    };
    let (rows, total) = resolved(&pool, "paper", &lost, 50, 0).await.unwrap();
    assert_eq!((rows.len(), total), (1, 1));

    let b = breakdown(&pool, "paper", &HistoryFilter::default())
        .await
        .unwrap();
    let trading = b.iter().find(|r| r.bot == "trading").unwrap();
    assert_eq!(trading.n, 2);
    assert_eq!(trading.wins, 1);
    assert!((trading.pnl - (-10.0)).abs() < 1e-9);
}

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn empty_history(pool: PgPool) {
    let (rows, total) = resolved(&pool, "paper", &HistoryFilter::default(), 50, 0)
        .await
        .unwrap();
    assert!(rows.is_empty());
    assert_eq!(total, 0);
    assert!(
        breakdown(&pool, "paper", &HistoryFilter::default())
            .await
            .unwrap()
            .is_empty()
    );
}
