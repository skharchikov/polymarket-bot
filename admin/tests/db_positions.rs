use admin::db::positions::{Filter, detail, open_positions};
use sqlx::PgPool;

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn filters_and_detail(pool: PgPool) {
    sqlx::query(
        "INSERT INTO bets (id, market_id, question, side, entry_price, slipped_price, shares, cost, fee_paid, \
         estimated_prob, confidence, edge, kelly_size, reasoning, strategy, source, copy_ref) VALUES \
         (1,'m1','Fed?','No',0.7,0.71,70,50,1,0.8,0.6,0.1,0.03,'xgb','balanced','xgboost',NULL), \
         (2,'m2','BTC?','Yes',0.4,0.41,25,10,0.2,0.5,0.5,0.1,0.02,'copy','copy:0xab','copy_trade','{\"trader\":\"alice | Rank #3 | PnL $9k\",\"wallet\":\"0xAB\",\"price\":0.4,\"size_usd\":100}')",
    )
    .execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO followed_traders (proxy_wallet, username) VALUES ('0xab', 'alice')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO bet_features (bet_id, features) VALUES (1, '{\"rsi_14\": 61.0}')")
        .execute(&pool)
        .await
        .unwrap();

    let all = open_positions(&pool, "paper", &Filter::default())
        .await
        .unwrap();
    assert_eq!(all.len(), 2);
    assert!(all.iter().find(|r| r.id == 1).unwrap().username.is_none());
    let copy = open_positions(
        &pool,
        "paper",
        &Filter {
            bot: Some("copy".into()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(copy.len(), 1);
    assert_eq!(copy[0].wallet.as_deref(), Some("0xAB"));
    assert_eq!(copy[0].username.as_deref(), Some("alice"));

    let d = detail(&pool, 1).await.unwrap().unwrap();
    assert!(d.features.unwrap().contains("rsi_14"));
    assert!(detail(&pool, 999).await.unwrap().is_none());
}
