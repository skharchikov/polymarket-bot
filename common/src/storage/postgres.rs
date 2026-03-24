use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::PgPool;

use super::portfolio::{Bet, BetContext, BetSide, CopyRef, NewBet};

/// A signal that was evaluated but didn't pass the scanner gate.
#[derive(Debug, Clone)]
pub struct RejectedSignal {
    pub market_id: String,
    pub question: String,
    pub reason: String,
    pub current_price: Option<f64>,
    pub estimated_prob: Option<f64>,
    pub edge: Option<f64>,
    pub confidence: Option<f64>,
    pub combined_lr: Option<f64>,
}

/// Returned from resolve_bet with all context needed for a rich Telegram message.
#[allow(dead_code)]
pub struct ResolvedBet {
    pub question: String,
    pub side: BetSide,
    pub won: bool,
    pub entry_price: f64,
    pub cost: f64,
    pub entry_fee: f64,
    pub shares: f64,
    pub edge: f64,
    pub confidence: f64,
    pub pnl: f64,
    pub bankroll: f64,
    pub total_wins: usize,
    pub total_losses: usize,
    pub total_pnl: f64,
    pub strat_wins: usize,
    pub strat_losses: usize,
    pub strat_pnl: f64,
    pub strategy: String,
    pub source: String,
    pub market_id: String,
}

/// Postgres-backed portfolio state.
pub struct PgPortfolio {
    pub(super) pool: PgPool,
}

impl PgPortfolio {
    pub async fn new(pool: PgPool) -> Result<Self> {
        Ok(Self { pool })
    }

    pub async fn run_migrations(&self) -> Result<()> {
        sqlx::migrate!("../migrations")
            .run(&self.pool)
            .await
            .context("failed to run migrations")?;
        Ok(())
    }

    /// Resolve LLM estimates when a market resolves (for calibration tracking).
    pub async fn resolve_estimates(&self, market_id: &str, yes_won: bool) -> Result<()> {
        sqlx::query(
            "UPDATE llm_estimates SET resolved = true, outcome = $1, resolved_at = NOW() \
             WHERE market_id = $2 AND resolved = false",
        )
        .bind(yes_won)
        .bind(market_id)
        .execute(&self.pool)
        .await?;
        // Also resolve prediction_log entries
        sqlx::query(
            "UPDATE prediction_log SET resolved = true, outcome = $1, resolved_at = NOW() \
             WHERE market_id = $2 AND resolved = false",
        )
        .bind(yes_won)
        .bind(market_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Log a model prediction for Brier score tracking.
    #[allow(clippy::too_many_arguments)]
    pub async fn log_prediction(
        &self,
        market_id: &str,
        source: &str,
        market_price: f64,
        model_prob: f64,
        posterior: f64,
        confidence: f64,
        edge: f64,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO prediction_log (market_id, source, market_price, model_prob, posterior, confidence, edge) \
             VALUES ($1,$2,$3,$4,$5,$6,$7)",
        )
        .bind(market_id)
        .bind(source)
        .bind(market_price)
        .bind(model_prob)
        .bind(posterior)
        .bind(confidence)
        .bind(edge)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Compute Brier score from resolved predictions.
    /// Returns (brier_score, n_predictions, market_brier) or None if no data.
    pub async fn brier_score(&self) -> Result<Option<(f64, usize, f64)>> {
        #[derive(sqlx::FromRow)]
        struct PredRow {
            posterior: f64,
            market_price: f64,
            outcome: Option<bool>,
        }

        let rows: Vec<PredRow> = sqlx::query_as(
            "SELECT posterior, market_price, outcome FROM prediction_log \
             WHERE resolved = true AND outcome IS NOT NULL AND source != 'copy_trade'",
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(None);
        }

        let n = rows.len();
        let mut model_sum = 0.0;
        let mut market_sum = 0.0;

        for r in &rows {
            let actual = if r.outcome.unwrap_or(false) { 1.0 } else { 0.0 };
            model_sum += (r.posterior - actual).powi(2);
            market_sum += (r.market_price - actual).powi(2);
        }

        Ok(Some((model_sum / n as f64, n, market_sum / n as f64)))
    }

    /// Market IDs rejected in the last `hours` — skip re-assessment.
    pub async fn recently_rejected_market_ids(&self, hours: i32) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT market_id FROM rejected_signals \
             WHERE created_at > NOW() - make_interval(hours => $1)",
        )
        .bind(hours)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    /// Persist a correlation-blocked signal so future checks treat it like an open bet.
    pub async fn save_correlation_blocked(
        &self,
        market_id: &str,
        question: &str,
        side: &BetSide,
        reason: &str,
        end_date: Option<&str>,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO correlation_blocked (market_id, question, side, reason, end_date) \
             VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(market_id)
        .bind(question)
        .bind(match side {
            BetSide::Yes => "Yes",
            BetSide::No => "No",
        })
        .bind(reason)
        .bind(end_date)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Fetch recent correlation-blocked signals that are still relevant:
    /// - has an end_date in the future, OR
    /// - no end_date but was blocked within the last 7 days.
    pub async fn recent_correlation_blocked(&self) -> Result<Vec<(String, BetSide)>> {
        let rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT question, side FROM correlation_blocked \
             WHERE (end_date IS NOT NULL AND end_date::timestamptz > NOW()) \
                OR (end_date IS NULL AND created_at > NOW() - INTERVAL '7 days')",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|(q, s)| {
                let side = if s == "No" { BetSide::No } else { BetSide::Yes };
                (q, side)
            })
            .collect())
    }

    /// Market IDs that already have resolved bets — avoid re-betting.
    pub async fn resolved_bet_market_ids(&self) -> Result<Vec<String>> {
        let rows: Vec<(String,)> =
            sqlx::query_as("SELECT DISTINCT market_id FROM bets WHERE resolved = true")
                .fetch_all(&self.pool)
                .await?;
        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    /// Persist rejected signals for post-hoc analysis.
    pub async fn save_rejected_signals(&self, rejections: &[RejectedSignal]) -> Result<()> {
        for r in rejections {
            sqlx::query(
                "INSERT INTO rejected_signals \
                 (market_id, question, reason, current_price, estimated_prob, edge, confidence, combined_lr) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            )
            .bind(&r.market_id)
            .bind(&r.question)
            .bind(&r.reason)
            .bind(r.current_price)
            .bind(r.estimated_prob)
            .bind(r.edge)
            .bind(r.confidence)
            .bind(r.combined_lr)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    /// Upsert a Telegram user (tracks who interacts with the bot).
    pub async fn upsert_telegram_user(
        &self,
        chat_id: &str,
        username: Option<&str>,
        first_name: Option<&str>,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO telegram_users (chat_id, username, first_name, last_seen) \
             VALUES ($1, $2, $3, NOW()) \
             ON CONFLICT (chat_id) DO UPDATE SET \
               username = COALESCE(EXCLUDED.username, telegram_users.username), \
               first_name = COALESCE(EXCLUDED.first_name, telegram_users.first_name), \
               last_seen = NOW()",
        )
        .bind(chat_id)
        .bind(username)
        .bind(first_name)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Get subscriber chat IDs with their usernames for logging.
    pub async fn telegram_subscribers(&self) -> Result<Vec<(String, Option<String>)>> {
        let rows: Vec<(String, Option<String>)> =
            sqlx::query_as("SELECT chat_id, username FROM telegram_users")
                .fetch_all(&self.pool)
                .await?;
        Ok(rows)
    }

    /// Build a stats summary string for /stats command.
    #[tracing::instrument(skip(self))]
    pub async fn stats_summary(&self) -> Result<String> {
        self.stats_summary_inner(true).await
    }

    pub async fn stats_summary_ml_only(&self) -> Result<String> {
        self.stats_summary_inner(false).await
    }

    async fn stats_summary_inner(&self, include_copy: bool) -> Result<String> {
        let starting = self.starting_bankroll().await?;
        let resolved = self.resolved_bets().await?;
        let open = self.open_bets().await?;

        // Read active strategies from portfolio (set by init_strategy_bankrolls)
        let strat_str = self.get_text("active_strategies").await?;
        let strategies: Vec<String> = if strat_str.is_empty() {
            vec![
                "aggressive".into(),
                "balanced".into(),
                "conservative".into(),
            ]
        } else {
            strat_str.split(',').map(|s| s.trim().to_string()).collect()
        };
        // Build per-strategy stats
        let mut strat_stats = Vec::new();
        let mut total_bankroll = 0.0_f64;
        let mut total_pnl = 0.0_f64;
        let mut total_wins = 0_usize;
        let mut total_losses = 0_usize;
        let mut total_open = 0_usize;

        for strat in &strategies {
            let s_resolved: Vec<_> = resolved.iter().filter(|b| &b.strategy == strat).collect();
            let s_open_count = open.iter().filter(|b| &b.strategy == strat).count();
            let s_wins = s_resolved.iter().filter(|b| b.won == Some(true)).count();
            let s_losses = s_resolved.iter().filter(|b| b.won == Some(false)).count();
            let s_pnl: f64 = s_resolved.iter().filter_map(|b| b.pnl).sum();
            let s_bankroll = self.strategy_bankroll(strat).await?;
            let s_starting = self.strategy_starting_bankroll(strat).await.unwrap_or(0.0);
            let s_roi = if s_starting > 0.0 {
                s_pnl / s_starting * 100.0
            } else {
                0.0
            };

            total_bankroll += s_bankroll;
            total_pnl += s_pnl;
            total_wins += s_wins;
            total_losses += s_losses;
            total_open += s_open_count;

            strat_stats.push(crate::format::StratStats {
                name: strat.clone(),
                bankroll: s_bankroll,
                roi: s_roi,
                pnl: s_pnl,
                wins: s_wins,
                losses: s_losses,
                open: s_open_count,
            });
        }

        // Per-source breakdown
        let all_resolved: Vec<&Bet> = resolved.iter().collect();
        let mut source_stats = Vec::new();
        for src in &["xgboost", "llm_consensus"] {
            let src_bets: Vec<_> = all_resolved.iter().filter(|b| b.source == *src).collect();
            if src_bets.is_empty() {
                continue;
            }
            let src_wins = src_bets.iter().filter(|b| b.won == Some(true)).count();
            let src_losses = src_bets.iter().filter(|b| b.won == Some(false)).count();
            let src_pnl: f64 = src_bets.iter().filter_map(|b| b.pnl).sum();
            source_stats.push(crate::format::SourceStats {
                name: src.to_string(),
                wins: src_wins,
                losses: src_losses,
                pnl: src_pnl,
            });
        }

        // Fetch live unrealized PnL from current market prices (split ML vs copy)
        let ((ml_unrealized, ml_exposure), (copy_unrealized, copy_exposure)) = if !open.is_empty() {
            self.live_unrealized().await
        } else {
            ((0.0, 0.0), (0.0, 0.0))
        };

        let copy_trade = if include_copy {
            // Copy-trade aggregate
            let active_traders = self.get_active_traders().await.unwrap_or_default();
            let copy_traders = active_traders.len();
            let mut copy_bankroll = 0.0_f64;
            for trader in &active_traders {
                let short = &trader.proxy_wallet[..8.min(trader.proxy_wallet.len())];
                let strat = format!("copy:{short}");
                copy_bankroll += self.strategy_bankroll(&strat).await.unwrap_or(0.0);
            }
            let copy_resolved: Vec<_> = resolved
                .iter()
                .filter(|b| b.strategy.starts_with("copy:"))
                .collect();
            let copy_open = open
                .iter()
                .filter(|b| b.strategy.starts_with("copy:"))
                .count();
            let copy_wins = copy_resolved.iter().filter(|b| b.won == Some(true)).count();
            let copy_losses = copy_resolved
                .iter()
                .filter(|b| b.won == Some(false))
                .count();
            let copy_pnl: f64 = copy_resolved.iter().filter_map(|b| b.pnl).sum();
            Some(crate::format::CopyTradeSummary {
                traders: copy_traders,
                open: copy_open,
                wins: copy_wins,
                losses: copy_losses,
                pnl: copy_pnl,
                bankroll: copy_bankroll,
                unrealized: copy_unrealized,
                exposure: copy_exposure,
            })
        } else {
            let _ = (copy_unrealized, copy_exposure);
            None
        };

        let stats_data = crate::format::StatsData {
            ml_bankroll: total_bankroll,
            starting,
            total_pnl,
            total_wins,
            total_losses,
            total_open,
            ml_unrealized,
            ml_exposure,
            strategies: strat_stats,
            sources: source_stats,
            copy_trade,
        };

        Ok(crate::format::format_stats(&stats_data))
    }

    /// Build a summary of open ML bets for /open command.
    #[tracing::instrument(skip(self))]
    pub async fn open_bets_summary(&self) -> Result<String> {
        self.open_bets_summary_filtered(false).await
    }

    /// Build a summary of open copy-trade positions for /positions command.
    #[tracing::instrument(skip(self))]
    pub async fn open_copy_summary(&self) -> Result<String> {
        self.open_bets_summary_filtered(true).await
    }

    async fn open_bets_summary_filtered(&self, copy_only: bool) -> Result<String> {
        use crate::data::models::fetch_yes_prices;
        use crate::format::{self, OpenBetView};

        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()?;
        let all_open = self.open_bets().await?;
        let open: Vec<_> = all_open
            .iter()
            .filter(|b| b.strategy.starts_with("copy:") == copy_only)
            .collect();

        let empty_msg = if copy_only {
            "📭 No open copy-trade positions"
        } else {
            "📭 No open model positions"
        };
        if open.is_empty() {
            return Ok(empty_msg.to_string());
        }

        let ids: Vec<&str> = open.iter().map(|b| b.market_id.as_str()).collect();
        let prices = fetch_yes_prices(&http, &ids).await;

        let views: Vec<OpenBetView> = open
            .iter()
            .zip(prices)
            .map(|(bet, yes_price)| {
                let poly_url = if bet.url.is_empty() {
                    None
                } else {
                    Some(bet.url.clone())
                };
                OpenBetView {
                    bet,
                    current_yes_price: yes_price,
                    poly_url,
                }
            })
            .collect();

        Ok(format::format_open_bets(&views, false))
    }

    /// Fetch live unrealized PnL and exposure, split into (ml, copy) tuples of (unrealized, exposure).
    pub(super) async fn live_unrealized(&self) -> ((f64, f64), (f64, f64)) {
        use crate::data::models::fetch_yes_prices;
        use crate::storage::portfolio::BetSide;

        let open = match self.open_bets().await {
            Ok(b) => b,
            Err(_) => return ((0.0, 0.0), (0.0, 0.0)),
        };
        if open.is_empty() {
            return ((0.0, 0.0), (0.0, 0.0));
        }

        let http = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
        {
            Ok(c) => c,
            Err(_) => return ((0.0, 0.0), (0.0, 0.0)),
        };

        let ids: Vec<&str> = open.iter().map(|b| b.market_id.as_str()).collect();
        let prices = fetch_yes_prices(&http, &ids).await;

        let mut ml_unrealized = 0.0_f64;
        let mut ml_exposure = 0.0_f64;
        let mut copy_unrealized = 0.0_f64;
        let mut copy_exposure = 0.0_f64;

        for (bet, yes_price) in open.iter().zip(prices) {
            let is_copy = bet.strategy.starts_with("copy:");
            if is_copy {
                copy_exposure += bet.cost;
            } else {
                ml_exposure += bet.cost;
            }
            if let Some(yp) = yes_price {
                let cur = match bet.side {
                    BetSide::Yes => yp,
                    BetSide::No => 1.0 - yp,
                };
                let pnl = bet.shares * cur - bet.cost;
                if is_copy {
                    copy_unrealized += pnl;
                } else {
                    ml_unrealized += pnl;
                }
            }
        }
        (
            (ml_unrealized, ml_exposure),
            (copy_unrealized, copy_exposure),
        )
    }

    /// Build a model accuracy summary for /brier command.
    #[tracing::instrument(skip(self))]
    pub async fn brier_summary(&self) -> Result<String> {
        #[derive(sqlx::FromRow)]
        struct PredRow {
            source: String,
            posterior: f64,
            market_price: f64,
            outcome: Option<bool>,
        }

        let rows: Vec<PredRow> = sqlx::query_as(
            "SELECT source, posterior, market_price, outcome FROM prediction_log \
             WHERE resolved = true AND outcome IS NOT NULL AND source != 'copy_trade'",
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok("🎯 *Model Accuracy*\n\nNo resolved predictions yet.".to_string());
        }

        let n = rows.len();
        let mut model_sum = 0.0_f64;
        let mut market_sum = 0.0_f64;

        for r in &rows {
            let actual = if r.outcome.unwrap_or(false) { 1.0 } else { 0.0 };
            model_sum += (r.posterior - actual).powi(2);
            market_sum += (r.market_price - actual).powi(2);
        }

        let model_brier = model_sum / n as f64;
        let market_brier = market_sum / n as f64;
        let skill = if market_brier > 0.0 {
            (1.0 - model_brier / market_brier) * 100.0
        } else {
            0.0
        };
        let skill_emoji = if skill > 0.0 { "✅" } else { "❌" };

        // Per-source breakdown
        let mut source_lines = String::new();
        for source in &["xgboost", "llm_consensus"] {
            let src_rows: Vec<&PredRow> = rows.iter().filter(|r| r.source == *source).collect();
            if src_rows.is_empty() {
                continue;
            }
            let sn = src_rows.len();
            let mut sm = 0.0_f64;
            let mut smk = 0.0_f64;
            for r in &src_rows {
                let actual = if r.outcome.unwrap_or(false) { 1.0 } else { 0.0 };
                sm += (r.posterior - actual).powi(2);
                smk += (r.market_price - actual).powi(2);
            }
            let sb = sm / sn as f64;
            let smb = smk / sn as f64;
            let ss = if smb > 0.0 {
                (1.0 - sb / smb) * 100.0
            } else {
                0.0
            };
            let icon = if *source == "xgboost" { "🤖" } else { "🧠" };
            let se = if ss > 0.0 { "✅" } else { "❌" };
            source_lines.push_str(&format!(
                "\n{icon} _{source}_: `{sb:.4}` ({sn}) {se} `{ss:+.1}%`"
            ));
        }

        Ok(format!(
            "🎯 *Model Accuracy*\n\n\
             📊 Predictions: {n}\n\
             🤖 Model Brier: `{model_brier:.4}` (lower = better)\n\
             📈 Market Brier: `{market_brier:.4}`\n\
             {skill_emoji} Skill vs market: `{skill:+.1}%`\n\
             {source_lines}\n\n\
             _Brier score: 0 = perfect, 0.25 = random_",
        ))
    }

    // --- meta helpers ---

    async fn get_f64(&self, key: &str) -> Result<f64> {
        let row: Option<(Option<f64>,)> =
            sqlx::query_as("SELECT value_f64 FROM portfolio WHERE key = $1")
                .bind(key)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.and_then(|r| r.0).unwrap_or(0.0))
    }

    async fn set_f64(&self, key: &str, val: f64) -> Result<()> {
        sqlx::query("UPDATE portfolio SET value_f64 = $1 WHERE key = $2")
            .bind(val)
            .bind(key)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn get_text(&self, key: &str) -> Result<String> {
        let row: Option<(Option<String>,)> =
            sqlx::query_as("SELECT value_text FROM portfolio WHERE key = $1")
                .bind(key)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.and_then(|r| r.0).unwrap_or_default())
    }

    async fn set_text(&self, key: &str, val: &str) -> Result<()> {
        sqlx::query("UPDATE portfolio SET value_text = $1 WHERE key = $2")
            .bind(val)
            .bind(key)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn upsert_f64_pub(&self, key: &str, val: f64) -> Result<()> {
        self.upsert_f64(key, val).await
    }

    async fn upsert_f64(&self, key: &str, val: f64) -> Result<()> {
        sqlx::query(
            "INSERT INTO portfolio (key, value_f64) VALUES ($1, $2) \
             ON CONFLICT (key) DO UPDATE SET value_f64 = EXCLUDED.value_f64",
        )
        .bind(key)
        .bind(val)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn upsert_text(&self, key: &str, val: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO portfolio (key, value_text) VALUES ($1, $2) \
             ON CONFLICT (key) DO UPDATE SET value_text = EXCLUDED.value_text",
        )
        .bind(key)
        .bind(val)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // --- public API matching PortfolioState ---

    pub async fn starting_bankroll(&self) -> Result<f64> {
        self.get_f64("starting_bankroll").await
    }

    pub async fn bankroll(&self) -> Result<f64> {
        self.get_f64("bankroll").await
    }

    /// Get bankroll for a specific strategy.
    pub async fn strategy_bankroll(&self, strategy: &str) -> Result<f64> {
        self.get_f64(&format!("bankroll:{strategy}")).await
    }

    /// Get the starting (initial) bankroll for a specific strategy.
    pub async fn strategy_starting_bankroll(&self, strategy: &str) -> Result<f64> {
        self.get_f64(&format!("starting_bankroll:{strategy}")).await
    }

    /// Ensure a portfolio key exists with a default value; no-op if it already exists.
    pub async fn ensure_key(&self, key: &str, default: f64) -> Result<()> {
        sqlx::query(
            "INSERT INTO portfolio (key, value_f64) VALUES ($1, $2) ON CONFLICT (key) DO NOTHING",
        )
        .bind(key)
        .bind(default)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Check if a portfolio key exists.
    async fn key_exists(&self, key: &str) -> Result<bool> {
        let row: Option<(i32,)> = sqlx::query_as("SELECT 1 FROM portfolio WHERE key = $1")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.is_some())
    }

    /// Initialize per-strategy bankroll keys if they don't exist.
    /// Each strategy gets `strategy_bankroll` EUR independently.
    /// `strategy_names` is a slice of strategy name strings (e.g. "aggressive").
    pub async fn init_strategy_bankrolls(
        &self,
        strategy_names: &[String],
        strategy_bankroll: f64,
    ) -> Result<()> {
        for name in strategy_names {
            let key = format!("bankroll:{name}");
            if !self.key_exists(&key).await? {
                self.upsert_f64(&key, strategy_bankroll).await?;
                tracing::info!(
                    strategy = %name,
                    bankroll = format_args!("€{strategy_bankroll:.2}"),
                    "Initialized strategy bankroll"
                );
            }
            let start_key = format!("starting_bankroll:{name}");
            if !self.key_exists(&start_key).await? {
                self.upsert_f64(&start_key, strategy_bankroll).await?;
            }
        }

        // Store active strategy names so /stats can show all of them
        let strat_names: Vec<&str> = strategy_names.iter().map(|s| s.as_str()).collect();
        self.upsert_text("active_strategies", &strat_names.join(","))
            .await?;

        // Set starting_bankroll only if not already stored (first run)
        if !self.key_exists("starting_bankroll").await? || self.starting_bankroll().await? == 0.0 {
            let total = strategy_names.len() as f64 * strategy_bankroll;
            self.upsert_f64("starting_bankroll", total).await?;
            tracing::info!(
                starting = format_args!("€{total:.2}"),
                "Set starting_bankroll (first run)"
            );
        }

        // Init signal counters (idempotent — won't overwrite existing)
        for name in strategy_names {
            let key = format!("signals_sent_today:{name}");
            if !self.key_exists(&key).await? {
                self.upsert_f64(&key, 0.0).await?;
            }
            let key = format!("last_signal_date:{name}");
            if !self.key_exists(&key).await? {
                self.upsert_text(&key, "").await?;
            }
        }

        Ok(())
    }

    pub async fn signals_sent_today(&self) -> Result<usize> {
        Ok(self.get_f64("signals_sent_today").await? as usize)
    }

    pub async fn strategy_signals_today(&self, strategy: &str) -> Result<usize> {
        Ok(self
            .get_f64(&format!("signals_sent_today:{strategy}"))
            .await? as usize)
    }

    pub async fn reset_daily_if_needed(&self) -> Result<()> {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let last = self.get_text("last_signal_date").await?;
        if last != today {
            self.set_f64("signals_sent_today", 0.0).await?;
            self.set_text("last_signal_date", &today).await?;
        }
        Ok(())
    }

    pub async fn reset_strategy_daily_if_needed(&self, strategy: &str) -> Result<()> {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let last = self
            .get_text(&format!("last_signal_date:{strategy}"))
            .await?;
        if last != today {
            self.set_f64(&format!("signals_sent_today:{strategy}"), 0.0)
                .await?;
            self.set_text(&format!("last_signal_date:{strategy}"), &today)
                .await?;
        }
        Ok(())
    }

    #[tracing::instrument(skip(self, bet), fields(market_id = %bet.market_id, strategy = %bet.strategy))]
    pub async fn place_bet(&self, bet: &NewBet) -> Result<i32> {
        let side_str = match bet.side {
            BetSide::Yes => "Yes",
            BetSide::No => "No",
        };
        let ctx_json = bet
            .context
            .as_ref()
            .map(|c| serde_json::to_value(c).unwrap_or_default());

        let mut tx = self.pool.begin().await?;

        let copy_ref_json = bet
            .copy_ref
            .as_ref()
            .map(|c| serde_json::to_value(c).unwrap_or_default());

        let row: (i32,) = sqlx::query_as(
            "INSERT INTO bets (market_id, question, side, entry_price, slipped_price, shares, cost, fee_paid, \
             estimated_prob, confidence, edge, kelly_size, reasoning, end_date, context, strategy, source, url, event_slug, copy_ref) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20) RETURNING id",
        )
        .bind(&bet.market_id)
        .bind(&bet.question)
        .bind(side_str)
        .bind(bet.entry_price)
        .bind(bet.slipped_price)
        .bind(bet.shares)
        .bind(bet.cost)
        .bind(bet.fee)
        .bind(bet.estimated_prob)
        .bind(bet.confidence)
        .bind(bet.edge)
        .bind(bet.kelly_size)
        .bind(&bet.reasoning)
        .bind(bet.end_date.as_deref())
        .bind(ctx_json)
        .bind(&bet.strategy)
        .bind(&bet.source)
        .bind(&bet.url)
        .bind(bet.event_slug.as_deref())
        .bind(copy_ref_json)
        .fetch_one(&mut *tx)
        .await?;

        // Batch update: deduct bankrolls and increment signal counters in one query
        let strat_bankroll_key = format!("bankroll:{}", bet.strategy);
        let strat_signals_key = format!("signals_sent_today:{}", bet.strategy);
        let total_deduction = bet.cost + bet.fee;
        sqlx::query(
            "UPDATE portfolio SET \
               value_f64 = CASE \
                 WHEN key IN ($1, 'bankroll') THEN value_f64 - $2 \
                 WHEN key IN ($3, 'signals_sent_today') THEN value_f64 + 1 \
               END \
             WHERE key IN ($1, 'bankroll', $3, 'signals_sent_today')",
        )
        .bind(&strat_bankroll_key)
        .bind(total_deduction)
        .bind(&strat_signals_key)
        .execute(&mut *tx)
        .await?;

        // Store feature vector for online learning
        if let Some(features) = &bet.features {
            let features_json = serde_json::to_value(features).unwrap_or_default();
            sqlx::query(
                "INSERT INTO bet_features (bet_id, features) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            )
            .bind(row.0)
            .bind(features_json)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;

        Ok(row.0)
    }

    #[tracing::instrument(skip(self))]
    pub async fn resolve_bet(&self, market_id: &str, yes_won: bool) -> Result<Option<ResolvedBet>> {
        // Find unresolved bet for this market
        #[derive(sqlx::FromRow)]
        struct OpenBetRow {
            id: i32,
            side: String,
            shares: f64,
            cost: f64,
            fee_paid: f64,
            question: String,
            entry_price: f64,
            edge: f64,
            confidence: f64,
            strategy: String,
            source: String,
        }

        let row: Option<OpenBetRow> = sqlx::query_as(
            "SELECT id, side, shares, cost, fee_paid, question, entry_price, edge, confidence, \
             strategy, source FROM bets WHERE market_id = $1 AND resolved = false LIMIT 1",
        )
        .bind(market_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(r) = row else {
            return Ok(None);
        };
        let (
            bet_id,
            side_str,
            shares,
            cost,
            entry_fee,
            question,
            entry_price,
            edge,
            confidence,
            strategy,
            source,
        ) = (
            r.id,
            r.side,
            r.shares,
            r.cost,
            r.fee_paid,
            r.question,
            r.entry_price,
            r.edge,
            r.confidence,
            r.strategy,
            r.source,
        );

        let side = if side_str == "Yes" {
            BetSide::Yes
        } else {
            BetSide::No
        };
        let bet_won = match side {
            BetSide::Yes => yes_won,
            BetSide::No => !yes_won,
        };

        let fee_pct = 0.02;
        let gross_payout = if bet_won { shares } else { 0.0 };
        let exit_fee = gross_payout * fee_pct;
        let net_payout = gross_payout - exit_fee;
        // PnL includes both entry and exit fees for accurate accounting
        let pnl = net_payout - cost - entry_fee;

        let strat_bankroll_key = format!("bankroll:{strategy}");

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "UPDATE bets SET resolved = true, won = $1, pnl = $2, fee_paid = fee_paid + $3, \
             resolved_at = NOW() WHERE id = $4",
        )
        .bind(bet_won)
        .bind(pnl)
        .bind(exit_fee)
        .bind(bet_id)
        .execute(&mut *tx)
        .await?;

        // Atomically credit strategy bankroll
        sqlx::query("UPDATE portfolio SET value_f64 = value_f64 + $1 WHERE key = $2")
            .bind(net_payout)
            .bind(&strat_bankroll_key)
            .execute(&mut *tx)
            .await?;

        // Atomically credit global bankroll
        sqlx::query("UPDATE portfolio SET value_f64 = value_f64 + $1 WHERE key = 'bankroll'")
            .bind(net_payout)
            .execute(&mut *tx)
            .await?;

        // Read the updated strategy bankroll within the transaction for the return value
        let updated_strat_bankroll: Option<(Option<f64>,)> =
            sqlx::query_as("SELECT value_f64 FROM portfolio WHERE key = $1")
                .bind(&strat_bankroll_key)
                .fetch_optional(&mut *tx)
                .await?;
        let strat_bankroll = updated_strat_bankroll.and_then(|r| r.0).unwrap_or(0.0);

        tx.commit().await?;

        // Gather cumulative stats for the message
        let resolved = self.resolved_bets().await?;
        let wins = resolved.iter().filter(|b| b.won == Some(true)).count();
        let losses = resolved.iter().filter(|b| b.won == Some(false)).count();
        let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();
        let strat_resolved: Vec<_> = resolved.iter().filter(|b| b.strategy == strategy).collect();
        let strat_wins = strat_resolved
            .iter()
            .filter(|b| b.won == Some(true))
            .count();
        let strat_losses = strat_resolved
            .iter()
            .filter(|b| b.won == Some(false))
            .count();
        let strat_pnl: f64 = strat_resolved.iter().filter_map(|b| b.pnl).sum();

        Ok(Some(ResolvedBet {
            question,
            side,
            won: bet_won,
            entry_price,
            cost,
            entry_fee,
            shares,
            edge,
            confidence,
            pnl,
            bankroll: strat_bankroll,
            total_wins: wins,
            total_losses: losses,
            total_pnl,
            strat_wins,
            strat_losses,
            strat_pnl,
            strategy,
            source,
            market_id: market_id.to_string(),
        }))
    }

    /// Early exit: sell a position at current market price (paper trade).
    /// Returns the same ResolvedBet struct as resolve_bet for consistent messaging.
    #[tracing::instrument(skip(self))]
    pub async fn early_exit(
        &self,
        bet_id: i32,
        current_yes_price: f64,
        reason: &str,
    ) -> Result<Option<ResolvedBet>> {
        #[derive(sqlx::FromRow)]
        struct OpenBetRow {
            id: i32,
            market_id: String,
            side: String,
            shares: f64,
            cost: f64,
            fee_paid: f64,
            question: String,
            entry_price: f64,
            edge: f64,
            confidence: f64,
            strategy: String,
            source: String,
        }

        let row: Option<OpenBetRow> = sqlx::query_as(
            "SELECT id, market_id, side, shares, cost, fee_paid, question, entry_price, edge, \
             confidence, strategy, source FROM bets WHERE id = $1 AND resolved = false LIMIT 1",
        )
        .bind(bet_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(r) = row else {
            return Ok(None);
        };

        let side = if r.side == "Yes" {
            BetSide::Yes
        } else {
            BetSide::No
        };

        // Selling shares at current market price
        let sell_price = match side {
            BetSide::Yes => current_yes_price,
            BetSide::No => 1.0 - current_yes_price,
        };
        let fee_pct = 0.02;
        let gross_payout = r.shares * sell_price;
        let exit_fee = gross_payout * fee_pct;
        let net_payout = gross_payout - exit_fee;
        let pnl = net_payout - r.cost - r.fee_paid;

        let bet_won = pnl > 0.0;
        let strat_bankroll_key = format!("bankroll:{}", r.strategy);

        let reasoning = format!("Early exit: {reason}");

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "UPDATE bets SET resolved = true, won = $1, pnl = $2, fee_paid = fee_paid + $3, \
             reasoning = reasoning || E'\\n' || $4, resolved_at = NOW() WHERE id = $5",
        )
        .bind(bet_won)
        .bind(pnl)
        .bind(exit_fee)
        .bind(&reasoning)
        .bind(r.id)
        .execute(&mut *tx)
        .await?;

        // Atomically credit strategy bankroll with sell proceeds
        sqlx::query("UPDATE portfolio SET value_f64 = value_f64 + $1 WHERE key = $2")
            .bind(net_payout)
            .bind(&strat_bankroll_key)
            .execute(&mut *tx)
            .await?;

        // Atomically credit global bankroll
        sqlx::query("UPDATE portfolio SET value_f64 = value_f64 + $1 WHERE key = 'bankroll'")
            .bind(net_payout)
            .execute(&mut *tx)
            .await?;

        // Read the updated strategy bankroll within the transaction for the return value
        let updated_strat_bankroll: Option<(Option<f64>,)> =
            sqlx::query_as("SELECT value_f64 FROM portfolio WHERE key = $1")
                .bind(&strat_bankroll_key)
                .fetch_optional(&mut *tx)
                .await?;
        let strat_bankroll = updated_strat_bankroll.and_then(|r| r.0).unwrap_or(0.0);

        tx.commit().await?;

        let resolved = self.resolved_bets().await?;
        let wins = resolved.iter().filter(|b| b.won == Some(true)).count();
        let losses = resolved.iter().filter(|b| b.won == Some(false)).count();
        let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();
        let strat_resolved: Vec<_> = resolved
            .iter()
            .filter(|b| b.strategy == r.strategy)
            .collect();
        let strat_wins = strat_resolved
            .iter()
            .filter(|b| b.won == Some(true))
            .count();
        let strat_losses = strat_resolved
            .iter()
            .filter(|b| b.won == Some(false))
            .count();
        let strat_pnl: f64 = strat_resolved.iter().filter_map(|b| b.pnl).sum();

        Ok(Some(ResolvedBet {
            question: r.question,
            side,
            won: bet_won,
            entry_price: r.entry_price,
            cost: r.cost,
            entry_fee: r.fee_paid,
            shares: r.shares,
            edge: r.edge,
            confidence: r.confidence,
            pnl,
            bankroll: strat_bankroll,
            total_wins: wins,
            total_losses: losses,
            total_pnl,
            strat_wins,
            strat_losses,
            strat_pnl,
            strategy: r.strategy,
            source: r.source,
            market_id: r.market_id,
        }))
    }

    /// Backfill missing URLs for existing bets from Gamma API.
    pub async fn backfill_urls(&self) -> Result<usize> {
        let rows: Vec<(i32, String)> =
            sqlx::query_as("SELECT id, market_id FROM bets WHERE url = '' OR url IS NULL")
                .fetch_all(&self.pool)
                .await?;
        if rows.is_empty() {
            return Ok(0);
        }

        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()?;

        let mut filled = 0usize;
        for (id, market_id) in &rows {
            let url: Option<String> = async {
                let api_url = format!("https://gamma-api.polymarket.com/markets/{market_id}");
                let text = http.get(&api_url).send().await.ok()?.text().await.ok()?;
                let v: serde_json::Value = serde_json::from_str(&text).ok()?;
                let event_slug = v["events"]
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(|e| e["slug"].as_str());
                let market_slug = v["slug"].as_str();
                match event_slug {
                    Some(ev) => match market_slug {
                        Some(mk) if mk != ev => {
                            Some(format!("https://polymarket.com/event/{ev}/{mk}"))
                        }
                        _ => Some(format!("https://polymarket.com/event/{ev}")),
                    },
                    None => None,
                }
            }
            .await;

            if let Some(url) = url {
                sqlx::query("UPDATE bets SET url = $1 WHERE id = $2")
                    .bind(&url)
                    .bind(id)
                    .execute(&self.pool)
                    .await?;
                filled += 1;
            }
        }

        tracing::info!(total = rows.len(), filled, "Backfilled bet URLs");
        Ok(filled)
    }

    pub async fn open_bet_market_ids(&self) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT market_id FROM bets WHERE resolved = false AND strategy NOT LIKE 'copy:%'",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    pub async fn open_copy_bet_market_ids(&self) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT market_id FROM bets WHERE resolved = false AND strategy LIKE 'copy:%'",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    pub async fn open_bet_event_slugs(&self) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT event_slug FROM bets WHERE resolved = false AND event_slug IS NOT NULL",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    /// Event slugs with open ML-only bets (excludes copy-trade bets).
    /// Used by the ML bet scan so copy positions don't block new model signals.
    pub async fn open_ml_bet_event_slugs(&self) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT event_slug FROM bets \
             WHERE resolved = false AND event_slug IS NOT NULL \
             AND strategy NOT LIKE 'copy:%'",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    pub async fn open_bets(&self) -> Result<Vec<Bet>> {
        self.fetch_bets("SELECT * FROM bets WHERE resolved = false ORDER BY placed_at DESC")
            .await
    }

    pub async fn resolved_bets(&self) -> Result<Vec<Bet>> {
        self.fetch_bets("SELECT * FROM bets WHERE resolved = true ORDER BY placed_at DESC")
            .await
    }

    pub async fn all_bets(&self) -> Result<Vec<Bet>> {
        self.fetch_bets("SELECT * FROM bets ORDER BY placed_at DESC")
            .await
    }

    async fn fetch_bets(&self, query: &str) -> Result<Vec<Bet>> {
        let rows: Vec<BetRow> = sqlx::query_as(query).fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(|r| r.into_bet()).collect())
    }

    pub async fn should_send_daily_report(&self) -> Result<bool> {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let last = self.get_text("last_daily_report_date").await?;
        Ok(last != today)
    }

    pub async fn mark_daily_report_sent(&self) -> Result<()> {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        self.set_text("last_daily_report_date", &today).await
    }

    pub async fn take_snapshot(&self) -> Result<()> {
        #[derive(sqlx::FromRow)]
        struct SnapshotAgg {
            total: i64,
            open_count: i64,
            wins: i64,
            losses: i64,
            total_pnl: f64,
        }
        let agg: SnapshotAgg = sqlx::query_as(
            "SELECT \
               COUNT(*) AS total, \
               COUNT(*) FILTER (WHERE NOT resolved) AS open_count, \
               COUNT(*) FILTER (WHERE won = true) AS wins, \
               COUNT(*) FILTER (WHERE won = false) AS losses, \
               COALESCE(SUM(pnl) FILTER (WHERE resolved), 0) AS total_pnl \
             FROM bets",
        )
        .fetch_one(&self.pool)
        .await?;

        let starting = self.starting_bankroll().await?;
        let bankroll = self.bankroll().await?;
        let roi = if starting > 0.0 {
            (bankroll - starting) / starting * 100.0
        } else {
            0.0
        };
        let today = Utc::now().format("%Y-%m-%d").to_string();

        sqlx::query(
            "INSERT INTO daily_snapshots (date, bankroll, open_bets, total_bets, wins, losses, total_pnl, roi_pct) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
        )
        .bind(&today)
        .bind(bankroll)
        .bind(agg.open_count as i32)
        .bind(agg.total as i32)
        .bind(agg.wins as i32)
        .bind(agg.losses as i32)
        .bind(agg.total_pnl)
        .bind(roi)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Generate daily stats summary — same format as before.
    #[tracing::instrument(skip(self))]
    pub async fn daily_summary(&self) -> Result<String> {
        let bankroll = self.bankroll().await?;
        let starting = self.starting_bankroll().await?;

        // Single aggregate query instead of 3 separate SELECT * fetches
        #[derive(sqlx::FromRow)]
        struct DailyAgg {
            total: i64,
            resolved_count: i64,
            wins: i64,
            losses: i64,
            total_pnl: f64,
            total_fees: f64,
            open_count: i64,
            open_exposure: f64,
        }
        let agg: DailyAgg = sqlx::query_as(
            "SELECT \
               COUNT(*) AS total, \
               COUNT(*) FILTER (WHERE resolved) AS resolved_count, \
               COUNT(*) FILTER (WHERE won = true) AS wins, \
               COUNT(*) FILTER (WHERE won = false) AS losses, \
               COALESCE(SUM(pnl) FILTER (WHERE resolved), 0) AS total_pnl, \
               COALESCE(SUM(fee_paid), 0) AS total_fees, \
               COUNT(*) FILTER (WHERE NOT resolved) AS open_count, \
               COALESCE(SUM(cost) FILTER (WHERE NOT resolved), 0) AS open_exposure \
             FROM bets",
        )
        .fetch_one(&self.pool)
        .await?;

        let roi = if starting > 0.0 {
            (bankroll - starting) / starting * 100.0
        } else {
            0.0
        };
        let win_rate = if agg.wins + agg.losses > 0 {
            agg.wins as f64 / (agg.wins + agg.losses) as f64 * 100.0
        } else {
            0.0
        };

        let mut msg = format!(
            "\u{1f4ca} *Daily Portfolio Report*\n\n\
             \u{1f4b0} Bankroll: `\u{20ac}{bankroll:.2}`\n\
             \u{1f4c8} Starting: `\u{20ac}{starting:.2}`\n\
             \u{1f4b5} Total PnL: `\u{20ac}{total_pnl:+.2}`\n\
             \u{1f4c9} Total fees: `\u{20ac}{total_fees:.2}`\n\
             \u{1f3af} ROI: `{roi:+.1}%`\n\n\
             \u{1f4cb} *Stats:*\n\
             Total bets: {total}\n\
             Resolved: {resolved_count} (\u{2705} {wins}W / \u{274c} {losses}L)\n\
             Win rate: {win_rate:.0}%\n\
             Open: {open_count} (\u{20ac}{open_exposure:.2} exposed)\n",
            total_pnl = agg.total_pnl,
            total_fees = agg.total_fees,
            total = agg.total,
            resolved_count = agg.resolved_count,
            wins = agg.wins,
            losses = agg.losses,
            open_count = agg.open_count,
            open_exposure = agg.open_exposure,
        );

        // Brier score — model accuracy vs market
        if let Ok(Some((brier, n_preds, market_brier))) = self.brier_score().await {
            let skill = if market_brier > 0.0 {
                (1.0 - brier / market_brier) * 100.0
            } else {
                0.0
            };
            msg.push_str(&format!(
                "\n\u{1f3af} *Model accuracy ({n_preds} predictions):*\n\
                 Brier score: `{brier:.4}` (market: `{market_brier:.4}`)\n\
                 Skill vs market: `{skill:+.1}%`\n",
            ));
        }

        // Only fetch open bets for the listing (lightweight — typically <20 rows)
        if agg.open_count > 0 {
            #[derive(sqlx::FromRow)]
            struct OpenBetBrief {
                question: String,
                side: String,
                cost: f64,
                slipped_price: f64,
                edge: f64,
            }
            let open_list: Vec<OpenBetBrief> = sqlx::query_as(
                "SELECT question, side, cost, slipped_price, edge \
                 FROM bets WHERE NOT resolved ORDER BY placed_at DESC",
            )
            .fetch_all(&self.pool)
            .await?;

            msg.push_str("\n\u{1f513} *Open bets:*\n");
            for bet in &open_list {
                let side_str = if bet.side == "No" { "NO" } else { "YES" };
                msg.push_str(&format!(
                    "\u{2022} _{question}_ \u{2014} {side} \u{20ac}{cost:.2} @ {price:.0}\u{00a2} (edge +{edge:.0}%)\n",
                    question = crate::format::truncate(&bet.question, 50),
                    side = side_str,
                    cost = bet.cost,
                    price = bet.slipped_price * 100.0,
                    edge = bet.edge * 100.0,
                ));
            }
        }

        Ok(msg)
    }

    /// Deep learning summary for LLM — delegates to the same logic from portfolio module.
    #[tracing::instrument(skip(self))]
    pub async fn learning_summary(&self) -> Result<String> {
        use super::portfolio::PortfolioState;
        // Only fetch resolved bets — learning analysis doesn't use open bets
        let bets = self.resolved_bets().await?;
        let starting = self.starting_bankroll().await?;
        let bankroll = self.bankroll().await?;
        let state = PortfolioState {
            starting_bankroll: starting,
            bankroll,
            bets,
            daily_snapshots: Vec::new(),
            signals_sent_today: 0,
            last_signal_date: String::new(),
            last_daily_report_date: String::new(),
        };
        Ok(state.learning_summary())
    }
}

// --- Row mapping ---

#[derive(sqlx::FromRow)]
struct BetRow {
    id: i32,
    market_id: String,
    question: String,
    side: String,
    entry_price: f64,
    slipped_price: f64,
    shares: f64,
    cost: f64,
    fee_paid: f64,
    estimated_prob: f64,
    confidence: f64,
    edge: f64,
    kelly_size: f64,
    reasoning: String,
    end_date: Option<String>,
    context: Option<serde_json::Value>,
    strategy: String,
    source: String,
    url: String,
    event_slug: Option<String>,
    placed_at: DateTime<Utc>,
    resolved: bool,
    won: Option<bool>,
    pnl: Option<f64>,
    resolved_at: Option<DateTime<Utc>>,
    copy_ref: Option<serde_json::Value>,
}

impl BetRow {
    fn into_bet(self) -> Bet {
        let side = if self.side == "No" {
            BetSide::No
        } else {
            BetSide::Yes
        };
        let context: Option<BetContext> = self.context.and_then(|v| serde_json::from_value(v).ok());
        let copy_ref: Option<CopyRef> = self.copy_ref.and_then(|v| serde_json::from_value(v).ok());

        Bet {
            id: self.id,
            market_id: self.market_id,
            question: self.question,
            side,
            entry_price: self.entry_price,
            slipped_price: self.slipped_price,
            shares: self.shares,
            cost: self.cost,
            fee_paid: self.fee_paid,
            estimated_prob: self.estimated_prob,
            confidence: self.confidence,
            edge: self.edge,
            kelly_size: self.kelly_size,
            reasoning: self.reasoning,
            end_date: self.end_date,
            context,
            strategy: self.strategy,
            source: self.source,
            url: self.url,
            event_slug: self.event_slug,
            placed_at: self.placed_at,
            resolved: self.resolved,
            won: self.won,
            pnl: self.pnl,
            resolved_at: self.resolved_at,
            copy_ref,
        }
    }
}

// --- Copy trading structs ---
// Not yet called from main — will be integrated in a later phase.

/// A trader we are following for copy-trade signals.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FollowedTrader {
    pub id: i32,
    pub proxy_wallet: String,
    pub username: Option<String>,
    pub source: String,
    pub rank: Option<i32>,
    pub pnl: Option<f64>,
    pub volume: Option<f64>,
    pub win_rate: Option<f64>,
    pub added_at: DateTime<Utc>,
    pub last_checked_at: Option<DateTime<Utc>>,
    pub active: bool,
}

/// An event row already persisted to `copy_trade_events`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CopyTradeEvent {
    pub id: i32,
    pub trader_wallet: String,
    pub market_id: String,
    pub condition_id: String,
    pub side: String,
    pub price: f64,
    pub size_usd: f64,
    pub tx_hash: Option<String>,
    pub detected_at: DateTime<Utc>,
    pub acted_on: bool,
    pub skip_reason: Option<String>,
}

/// Data needed to insert a new copy-trade event.
#[derive(Debug, Clone)]
pub struct NewCopyTradeEvent {
    pub trader_wallet: String,
    pub market_id: String,
    pub condition_id: String,
    pub side: String,
    pub price: f64,
    pub size_usd: f64,
    pub tx_hash: Option<String>,
}

// --- Copy trading DB row type ---

#[derive(sqlx::FromRow)]
pub(super) struct FollowedTraderRow {
    id: i32,
    proxy_wallet: String,
    username: Option<String>,
    source: String,
    rank: Option<i32>,
    pnl: Option<f64>,
    volume: Option<f64>,
    win_rate: Option<f64>,
    added_at: DateTime<Utc>,
    last_checked_at: Option<DateTime<Utc>>,
    active: bool,
}

impl FollowedTraderRow {
    pub(super) fn into_trader(self) -> FollowedTrader {
        FollowedTrader {
            id: self.id,
            proxy_wallet: self.proxy_wallet,
            username: self.username,
            source: self.source,
            rank: self.rank,
            pnl: self.pnl,
            volume: self.volume,
            win_rate: self.win_rate,
            added_at: self.added_at,
            last_checked_at: self.last_checked_at,
            active: self.active,
        }
    }
}
