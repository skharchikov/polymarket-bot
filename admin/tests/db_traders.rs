use admin::db::traders::traders;
use sqlx::PgPool;

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn joins_copy_pnl_by_wallet_case_insensitively(pool: PgPool) {
    sqlx::query("INSERT INTO followed_traders (proxy_wallet, username, rank, pnl, pnl_month) VALUES ('0xabcdef', 'alice', 3, 120000, 9000), ('0x999999', 'bob', 10, 5, 1)")
        .execute(&pool).await.unwrap();
    sqlx::query(
        "INSERT INTO bets (market_id, question, entry_price, slipped_price, shares, cost, fee_paid, estimated_prob, confidence, edge, kelly_size, reasoning, strategy, source, copy_ref, resolved, won, pnl) VALUES \
         ('m1','Q',0.5,0.5,10,20,0.4,0.6,0.5,0.1,0.05,'r','copy:0xab','copy_trade','{\"trader\":\"alice\",\"wallet\":\"0xABCDEF\",\"price\":0.5,\"size_usd\":50}', true, true, 8.0), \
         ('m2','Q',0.5,0.5,10,20,0.4,0.6,0.5,0.1,0.05,'r','copy:0xab','copy_trade','{\"trader\":\"alice\",\"wallet\":\"0xabcdef\",\"price\":0.5,\"size_usd\":50}', false, NULL, NULL)",
    )
    .execute(&pool).await.unwrap();

    let t = traders(&pool, "paper").await.unwrap();
    let alice = t
        .iter()
        .find(|r| r.username.as_deref() == Some("alice"))
        .unwrap();
    assert_eq!(alice.our_bets, 2);
    assert_eq!(alice.our_open, 1);
    assert!((alice.our_pnl - 8.0).abs() < 1e-9);
    let bob = t
        .iter()
        .find(|r| r.username.as_deref() == Some("bob"))
        .unwrap();
    assert_eq!(bob.our_bets, 0);
}
