use admin::db::overview::{activity, equity_rows, kpis};
use sqlx::PgPool;

async fn bet(
    pool: &PgPool,
    strategy: &str,
    cost: f64,
    fee: f64,
    resolved: Option<(bool, f64)>,
    mode: &str,
) {
    sqlx::query(
        "INSERT INTO bets (market_id, question, entry_price, slipped_price, shares, cost, fee_paid, \
         estimated_prob, confidence, edge, kelly_size, reasoning, strategy, mode, resolved, won, pnl, resolved_at) \
         VALUES ('m','Q?',0.5,0.5,10,$1,$2,0.6,0.5,0.1,0.05,'r',$3,$4,$5,$6,$7, CASE WHEN $5 THEN NOW() END)",
    )
    .bind(cost).bind(fee).bind(strategy).bind(mode)
    .bind(resolved.is_some())
    .bind(resolved.map(|r| r.0))
    .bind(resolved.map(|r| r.1))
    .execute(pool).await.unwrap();
}

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn kpis_on_empty_db_are_zero(pool: PgPool) {
    let k = kpis(&pool, "paper").await.unwrap();
    assert_eq!(k.open_count, 0);
    assert_eq!(k.pnl_all, 0.0);
    assert!(k.roi_all_pct.is_none());
    assert!(k.win_rate_30d.is_none());
    assert!(equity_rows(&pool, "paper").await.unwrap().is_empty());
    assert!(activity(&pool, "paper").await.unwrap().is_empty());
}

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn kpis_use_net_over_gross_roi_and_respect_mode(pool: PgPool) {
    bet(&pool, "balanced", 100.0, 2.0, Some((true, 48.0)), "paper").await;
    bet(
        &pool,
        "copy:0xab…cd",
        50.0,
        1.0,
        Some((false, -51.0)),
        "paper",
    )
    .await;
    bet(&pool, "balanced", 30.0, 0.6, None, "paper").await;
    bet(&pool, "balanced", 999.0, 0.0, Some((true, 999.0)), "live").await; // must be excluded

    let k = kpis(&pool, "paper").await.unwrap();
    assert_eq!(k.open_count, 1);
    assert!((k.open_exposure - 30.6).abs() < 1e-9);
    assert!((k.pnl_all - (-3.0)).abs() < 1e-9);
    // ROI = -3 / (102 + 51) * 100
    assert!((k.roi_all_pct.unwrap() - (-3.0 / 153.0 * 100.0)).abs() < 1e-9);
    assert!((k.win_rate_30d.unwrap() - 50.0).abs() < 1e-9);

    let eq = equity_rows(&pool, "paper").await.unwrap();
    assert_eq!(eq.len(), 2);
}

#[sqlx::test(migrations = "../migrations")]
#[ignore = "needs DATABASE_URL"]
async fn bankroll_split_by_bot(pool: PgPool) {
    sqlx::query("INSERT INTO portfolio (key, value_f64) VALUES ('bankroll:balanced', 100), ('bankroll:conservative', 50), ('bankroll:copy:0xab', 25)")
        .execute(&pool).await.unwrap();
    let k = kpis(&pool, "paper").await.unwrap();
    assert_eq!(k.bankroll_trading, 150.0);
    assert_eq!(k.bankroll_copy, 25.0);
}
