//! Migration tests for the admin dashboard (bets.mode + admin_ro role).
//! Run with: DATABASE_URL=... cargo test -p polymarket-common -- --include-ignored

use sqlx::PgPool;

async fn insert_bet(pool: &PgPool) -> i32 {
    sqlx::query_scalar(
        "INSERT INTO bets (market_id, question, entry_price, slipped_price, shares, cost, fee_paid, \
         estimated_prob, confidence, edge, kelly_size, reasoning) \
         VALUES ('m1','Q?',0.5,0.5,10,5,0.1,0.6,0.5,0.1,0.05,'r') RETURNING id",
    )
    .fetch_one(pool)
    .await
    .unwrap()
}

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn bet_mode_defaults_to_paper(pool: PgPool) {
    let id = insert_bet(&pool).await;
    let mode: String = sqlx::query_scalar("SELECT mode FROM bets WHERE id = $1")
        .bind(id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(mode, "paper");
}

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn bet_mode_rejects_unknown_values(pool: PgPool) {
    let id = insert_bet(&pool).await;
    let res = sqlx::query("UPDATE bets SET mode = 'yolo' WHERE id = $1")
        .bind(id)
        .execute(&pool)
        .await;
    assert!(res.is_err(), "check constraint must reject 'yolo'");
}

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn admin_ro_can_read_but_not_write(pool: PgPool) {
    insert_bet(&pool).await;
    let mut conn = pool.acquire().await.unwrap();
    sqlx::query("SET ROLE admin_ro")
        .execute(&mut *conn)
        .await
        .unwrap();

    let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM bets")
        .fetch_one(&mut *conn)
        .await
        .unwrap();
    assert_eq!(n, 1);

    let write = sqlx::query("UPDATE bets SET question = 'x'")
        .execute(&mut *conn)
        .await;
    assert!(write.is_err(), "admin_ro must not write bets");

    let ins = sqlx::query("INSERT INTO portfolio (key, value_f64) VALUES ('k', 1)")
        .execute(&mut *conn)
        .await;
    assert!(ins.is_err(), "admin_ro must not write portfolio");
}
