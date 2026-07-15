//! Copy-trade database methods on [`PgPortfolio`].
//!
//! Lives in `src/storage/copy_trade.rs` as a sibling to `postgres.rs`.
//! Private postgres internals (`pool`, `FollowedTraderRow`) are exposed via
//! `pub(super)` so this module can access them without a full visibility bump.

use std::collections::HashMap;

use anyhow::{Context, Result};

use super::portfolio::Bet;
use super::postgres::{FollowedTrader, FollowedTraderRow, NewCopyTradeEvent, PgPortfolio};

impl PgPortfolio {
    /// Upsert a trader into `followed_traders`.
    ///
    /// On conflict (same `proxy_wallet`) the stats columns are refreshed and
    /// the trader is reactivated so stale deactivations are recovered.
    pub async fn add_followed_trader(
        &self,
        wallet: &str,
        username: Option<&str>,
        source: &str,
        rank: Option<i32>,
        pnl: Option<f64>,
        volume: Option<f64>,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO followed_traders \
               (proxy_wallet, username, source, rank, pnl, volume, active) \
             VALUES ($1, $2, $3, $4, $5, $6, TRUE) \
             ON CONFLICT (proxy_wallet) DO UPDATE SET \
               username   = COALESCE(EXCLUDED.username, followed_traders.username), \
               source     = EXCLUDED.source, \
               rank       = EXCLUDED.rank, \
               pnl        = EXCLUDED.pnl, \
               volume     = EXCLUDED.volume, \
               active     = TRUE",
        )
        .bind(wallet)
        .bind(username)
        .bind(source)
        .bind(rank)
        .bind(pnl)
        .bind(volume)
        .execute(&self.pool)
        .await
        .context("add_followed_trader")?;
        Ok(())
    }

    /// Return all traders with `active = true`.
    pub async fn get_active_traders(&self) -> Result<Vec<FollowedTrader>> {
        let rows: Vec<FollowedTraderRow> = sqlx::query_as(
            "SELECT id, proxy_wallet, username, source, rank, pnl, volume, win_rate, \
                    added_at, last_checked_at, active \
             FROM followed_traders \
             WHERE active = TRUE \
             ORDER BY rank ASC NULLS LAST",
        )
        .fetch_all(&self.pool)
        .await
        .context("get_active_traders")?;
        Ok(rows.into_iter().map(|r| r.into_trader()).collect())
    }

    /// Insert a new event into `copy_trade_events`.
    pub async fn save_copy_trade_event(&self, event: &NewCopyTradeEvent) -> Result<()> {
        sqlx::query(
            "INSERT INTO copy_trade_events \
               (trader_wallet, market_id, condition_id, side, price, size_usd, tx_hash) \
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(&event.trader_wallet)
        .bind(&event.market_id)
        .bind(&event.condition_id)
        .bind(&event.side)
        .bind(event.price)
        .bind(event.size_usd)
        .bind(&event.tx_hash)
        .execute(&self.pool)
        .await
        .context("save_copy_trade_event")?;
        Ok(())
    }

    /// Refresh leaderboard stats (rank, pnl, volume, username) for a trader.
    ///
    /// Unlike [`add_followed_trader`](Self::add_followed_trader) this does not
    /// touch `source` or `active`, so a deactivated trader stays deactivated.
    pub async fn update_trader_stats(
        &self,
        wallet: &str,
        rank: Option<i32>,
        pnl: Option<f64>,
        volume: Option<f64>,
        username: Option<&str>,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE followed_traders SET \
               rank     = $1, \
               pnl      = $2, \
               volume   = $3, \
               username = COALESCE($4, username) \
             WHERE proxy_wallet = $5",
        )
        .bind(rank)
        .bind(pnl)
        .bind(volume)
        .bind(username)
        .bind(wallet)
        .execute(&self.pool)
        .await
        .context("update_trader_stats")?;
        Ok(())
    }

    /// Set the `username` field for a trader (backfill / refresh).
    pub async fn update_trader_username(&self, wallet: &str, username: &str) -> Result<()> {
        sqlx::query("UPDATE followed_traders SET username = $1 WHERE proxy_wallet = $2")
            .bind(username)
            .bind(wallet)
            .execute(&self.pool)
            .await
            .context("update_trader_username")?;
        Ok(())
    }

    /// Stamp `last_checked_at = NOW()` for the given trader wallet.
    pub async fn update_trader_checked(&self, wallet: &str) -> Result<()> {
        sqlx::query("UPDATE followed_traders SET last_checked_at = NOW() WHERE proxy_wallet = $1")
            .bind(wallet)
            .execute(&self.pool)
            .await
            .context("update_trader_checked")?;
        Ok(())
    }

    /// Mark a trader as inactive so they are excluded from future polls.
    pub async fn deactivate_trader(&self, wallet: &str) -> Result<()> {
        sqlx::query("UPDATE followed_traders SET active = FALSE WHERE proxy_wallet = $1")
            .bind(wallet)
            .execute(&self.pool)
            .await
            .context("deactivate_trader")?;
        Ok(())
    }

    /// Return `true` if an identical trade (same trader, market, side, price) has
    /// already been recorded in `copy_trade_events`.  Used to avoid duplicates.
    pub async fn is_copy_trade_seen(
        &self,
        trader_wallet: &str,
        market_id: &str,
        side: &str,
        price: f64,
    ) -> Result<bool> {
        let row: Option<(i32,)> = sqlx::query_as(
            "SELECT id FROM copy_trade_events \
             WHERE trader_wallet = $1 AND market_id = $2 AND side = $3 AND price = $4 \
             LIMIT 1",
        )
        .bind(trader_wallet)
        .bind(market_id)
        .bind(side)
        .bind(price)
        .fetch_optional(&self.pool)
        .await
        .context("is_copy_trade_seen")?;
        Ok(row.is_some())
    }

    /// Look up a single followed trader by wallet address.
    pub async fn get_trader_by_wallet(&self, wallet: &str) -> Result<Option<FollowedTrader>> {
        let row: Option<FollowedTraderRow> = sqlx::query_as(
            "SELECT id, proxy_wallet, username, source, rank, pnl, volume, win_rate, \
                    added_at, last_checked_at, active \
             FROM followed_traders WHERE proxy_wallet = $1",
        )
        .bind(wallet)
        .fetch_optional(&self.pool)
        .await
        .context("get_trader_by_wallet")?;
        Ok(row.map(|r| r.into_trader()))
    }

    /// Get W/L/PnL record for bets placed under a copy-trader strategy.
    pub async fn copy_trader_record(&self, strategy_name: &str) -> Result<(usize, usize, f64)> {
        let rows: Vec<(Option<bool>, Option<f64>)> =
            sqlx::query_as("SELECT won, pnl FROM bets WHERE strategy = $1 AND resolved = true")
                .bind(strategy_name)
                .fetch_all(&self.pool)
                .await
                .context("copy_trader_record")?;

        let wins = rows.iter().filter(|(w, _)| *w == Some(true)).count();
        let losses = rows.iter().filter(|(w, _)| *w == Some(false)).count();
        let pnl: f64 = rows.iter().filter_map(|(_, p)| *p).sum();
        Ok((wins, losses, pnl))
    }

    /// Fetch per-trader stats rows for all active traders.
    ///
    /// Returns the structured [`TraderRow`] values alongside the raw open bets
    /// so callers can reuse the already-fetched data without a second DB round-trip.
    async fn collect_trader_rows(&self) -> Result<(Vec<crate::format::TraderRow>, Vec<Bet>)> {
        let traders = self.get_active_traders().await?;
        // Best-effort: open-bet fetch failures show 0 open bets rather than
        // failing the whole command (matches pre-refactor `/traders` behaviour).
        let open_bets = self.open_bets().await.unwrap_or_default();
        // Precompute open-bet counts per strategy in one pass (O(open_bets)).
        let open_by_strat: HashMap<&str, usize> =
            open_bets.iter().fold(HashMap::new(), |mut acc, b| {
                *acc.entry(b.strategy.as_str()).or_insert(0) += 1;
                acc
            });
        let mut rows = Vec::with_capacity(traders.len());
        for t in &traders {
            let short = &t.proxy_wallet[..8.min(t.proxy_wallet.len())];
            let name = t.username.as_deref().unwrap_or(short).to_string();
            let strat = format!("copy:{short}");
            let bankroll = self.strategy_bankroll(&strat).await.unwrap_or(0.0);
            let starting_bankroll = self.strategy_starting_bankroll(&strat).await.unwrap_or(0.0);
            let (wins, losses, pnl) = self.copy_trader_record(&strat).await.unwrap_or((0, 0, 0.0));
            let open_count = open_by_strat.get(strat.as_str()).copied().unwrap_or(0);
            rows.push(crate::format::TraderRow {
                name,
                wallet: t.proxy_wallet.clone(),
                wallet_short: short.to_string(),
                rank: t.rank,
                poly_pnl: t.pnl,
                bankroll,
                starting_bankroll,
                wins,
                losses,
                pnl,
                open: open_count,
            });
        }
        Ok((rows, open_bets))
    }

    /// Build aggregate copy-trading stats for the /stats command.
    pub async fn stats_summary_copy(&self) -> Result<String> {
        let (trader_rows, open_bets) = self.collect_trader_rows().await?;

        let total_bankroll: f64 = trader_rows.iter().map(|r| r.bankroll).sum();
        let total_starting: f64 = trader_rows.iter().map(|r| r.starting_bankroll).sum();
        let total_wins: usize = trader_rows.iter().map(|r| r.wins).sum();
        let total_losses: usize = trader_rows.iter().map(|r| r.losses).sum();
        let total_pnl: f64 = trader_rows.iter().map(|r| r.pnl).sum();

        let open_count = open_bets
            .iter()
            .filter(|b| b.strategy.starts_with("copy:"))
            .count();

        let (unrealized, exposure) = if open_count > 0 {
            let (_, copy) = self.live_unrealized().await;
            copy
        } else {
            (0.0, 0.0)
        };

        let data = crate::format::CopyStatsData {
            traders: trader_rows.len(),
            total_bankroll,
            starting_bankroll: total_starting,
            wins: total_wins,
            losses: total_losses,
            pnl: total_pnl,
            open: open_count,
            unrealized,
            exposure,
            trader_rows,
        };

        Ok(crate::format::format_copy_stats(&data))
    }

    /// Build a summary of followed traders for the /traders command.
    pub async fn traders_summary(&self) -> Result<String> {
        let (rows, _) = self.collect_trader_rows().await?;
        Ok(crate::format::format_traders(&rows))
    }
}
