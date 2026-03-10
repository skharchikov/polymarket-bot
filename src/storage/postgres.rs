use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::PgPool;

use crate::scanner::live::RejectedSignal;

use super::portfolio::{Bet, BetContext, BetSide, NewBet};

/// Returned from resolve_bet with all context needed for a rich Telegram message.
#[allow(dead_code)]
pub struct ResolvedBet {
    pub question: String,
    pub side: BetSide,
    pub won: bool,
    pub entry_price: f64,
    pub cost: f64,
    pub shares: f64,
    pub edge: f64,
    pub confidence: f64,
    pub pnl: f64,
    pub bankroll: f64,
    pub total_wins: usize,
    pub total_losses: usize,
    pub total_pnl: f64,
    pub strategy: String,
    pub source: String,
    pub market_id: String,
}

/// Postgres-backed portfolio state.
pub struct PgPortfolio {
    pool: PgPool,
}

impl PgPortfolio {
    pub async fn new(pool: PgPool) -> Result<Self> {
        Ok(Self { pool })
    }

    pub async fn run_migrations(&self) -> Result<()> {
        sqlx::migrate!("./migrations")
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
             WHERE resolved = true AND outcome IS NOT NULL",
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
        let num_strategies = strategies.len() as f64;
        let starting_per_strat = if num_strategies > 0.0 {
            starting / num_strategies
        } else {
            0.0
        };

        // Build per-strategy stats
        let mut strat_lines = Vec::new();
        let mut total_bankroll = 0.0_f64;
        let mut total_pnl = 0.0_f64;
        let mut total_wins = 0_usize;
        let mut total_losses = 0_usize;
        let mut total_open = 0_usize;

        for strat in &strategies {
            let label = match strat.as_str() {
                "aggressive" => "🔥",
                "balanced" => "⚖️",
                "conservative" => "🛡️",
                _ => "📊",
            };
            let s_resolved: Vec<_> = resolved.iter().filter(|b| &b.strategy == strat).collect();
            let s_open_count = open.iter().filter(|b| &b.strategy == strat).count();
            let s_wins = s_resolved.iter().filter(|b| b.won == Some(true)).count();
            let s_losses = s_resolved.iter().filter(|b| b.won == Some(false)).count();
            let s_pnl: f64 = s_resolved.iter().filter_map(|b| b.pnl).sum();
            let s_bankroll = self.strategy_bankroll(strat).await?;
            let s_roi = if starting_per_strat > 0.0 {
                (s_bankroll - starting_per_strat) / starting_per_strat * 100.0
            } else {
                0.0
            };
            let s_wr = if s_wins + s_losses > 0 {
                s_wins as f64 / (s_wins + s_losses) as f64 * 100.0
            } else {
                0.0
            };

            total_bankroll += s_bankroll;
            total_pnl += s_pnl;
            total_wins += s_wins;
            total_losses += s_losses;
            total_open += s_open_count;

            strat_lines.push(format!(
                "{label} *{strat}*\n\
                 \u{00a0}\u{00a0}💰 `€{s_bankroll:.2}` | ROI `{s_roi:+.1}%`\n\
                 \u{00a0}\u{00a0}💵 PnL `€{s_pnl:+.2}` | {s_wins}W/{s_losses}L ({s_wr:.0}%)\n\
                 \u{00a0}\u{00a0}🔓 {s_open_count} open",
            ));
        }

        let total_roi = if starting > 0.0 {
            (total_bankroll - starting) / starting * 100.0
        } else {
            0.0
        };
        let total_wr = if total_wins + total_losses > 0 {
            total_wins as f64 / (total_wins + total_losses) as f64 * 100.0
        } else {
            0.0
        };

        // Per-source breakdown
        let all_resolved: Vec<&Bet> = resolved.iter().collect();
        let mut source_lines = Vec::new();
        for src in &["xgboost", "llm_consensus"] {
            let src_bets: Vec<_> = all_resolved.iter().filter(|b| b.source == *src).collect();
            if src_bets.is_empty() {
                continue;
            }
            let src_wins = src_bets.iter().filter(|b| b.won == Some(true)).count();
            let src_losses = src_bets.iter().filter(|b| b.won == Some(false)).count();
            let src_pnl: f64 = src_bets.iter().filter_map(|b| b.pnl).sum();
            let src_wr = if src_wins + src_losses > 0 {
                src_wins as f64 / (src_wins + src_losses) as f64 * 100.0
            } else {
                0.0
            };
            let label = if *src == "xgboost" { "🤖" } else { "🧠" };
            source_lines.push(format!(
                "{label} *{src}*: {src_wins}W/{src_losses}L ({src_wr:.0}%) | PnL `€{src_pnl:+.2}`",
            ));
        }
        let source_section = if source_lines.is_empty() {
            String::new()
        } else {
            format!("\n\n📡 *By Source*\n{}", source_lines.join("\n"))
        };

        // Unrealized PnL (cached from housekeeping)
        let unrealized = self.get_f64("unrealized_pnl").await.unwrap_or(0.0);
        let exposure = self.get_f64("open_exposure").await.unwrap_or(0.0);
        let unrealized_section = if total_open > 0 {
            format!("\n📈 Unrealized: `€{unrealized:+.2}` (€{exposure:.2} deployed)\n")
        } else {
            String::new()
        };

        Ok(format!(
            "📊 *Bot Statistics*\n\n\
             💰 Bankroll: `€{total_bankroll:.2}` (started: `€{starting:.2}`)\n\
             💵 Realized PnL: `€{total_pnl:+.2}` | ROI: `{total_roi:+.1}%`\n\
             {unrealized_section}\
             📋 {total_wins}W / {total_losses}L ({total_wr:.0}%) | {total_open} open\n\n\
             {strat_details}{source_section}",
            strat_details = strat_lines.join("\n\n"),
        ))
    }

    /// Build a summary of open bets for /open command.
    #[tracing::instrument(skip(self))]
    pub async fn open_bets_summary(&self) -> Result<String> {
        use crate::format::{self, OpenBetView};

        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()?;
        let open = self.open_bets().await?;
        if open.is_empty() {
            return Ok("📭 No open bets".to_string());
        }

        let mut views = Vec::with_capacity(open.len());

        for bet in &open {
            let market_id = &bet.market_id;
            let market_data: Option<(f64, String)> = async {
                let url = format!("https://gamma-api.polymarket.com/markets/{market_id}");
                let text = http.get(&url).send().await.ok()?.text().await.ok()?;
                let v: serde_json::Value = serde_json::from_str(&text).ok()?;
                let prices_str = v["outcomePrices"].as_str()?;
                let prices: Vec<String> = serde_json::from_str(prices_str).ok()?;
                let yes_price = prices.first()?.parse::<f64>().ok()?;
                let event_slug = v["events"]
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(|e| e["slug"].as_str());
                let market_slug = v["slug"].as_str();
                let poly_url = match event_slug {
                    Some(ev) => match market_slug {
                        Some(mk) if mk != ev => {
                            format!("https://polymarket.com/event/{ev}/{mk}")
                        }
                        _ => format!("https://polymarket.com/event/{ev}"),
                    },
                    None => {
                        tracing::warn!(market_id, "Gamma API: no event_slug found");
                        String::new()
                    }
                };
                Some((yes_price, poly_url))
            }
            .await;

            if market_data.is_none() {
                tracing::warn!(market_id, question = %bet.question, "Failed to fetch market data from Gamma API");
            }

            let (price, url) = match market_data {
                Some((p, u)) => (Some(p), Some(u)),
                None => (None, None),
            };

            views.push(OpenBetView {
                bet,
                current_yes_price: price,
                poly_url: url,
            });
        }

        Ok(format::format_open_bets(&views, false))
    }

    /// Build a model accuracy summary for /brier command.
    #[tracing::instrument(skip(self))]
    pub async fn brier_summary(&self) -> Result<String> {
        match self.brier_score().await? {
            Some((model_brier, n, market_brier)) => {
                let skill = if market_brier > 0.0 {
                    (1.0 - model_brier / market_brier) * 100.0
                } else {
                    0.0
                };
                let skill_emoji = if skill > 0.0 { "✅" } else { "❌" };
                Ok(format!(
                    "🎯 *Model Accuracy*\n\n\
                     📊 Predictions: {n}\n\
                     🤖 Model Brier: `{model_brier:.4}` (lower = better)\n\
                     📈 Market Brier: `{market_brier:.4}`\n\
                     {skill_emoji} Skill vs market: `{skill:+.1}%`\n\n\
                     _Brier score: 0 = perfect, 0.25 = random_",
                ))
            }
            None => Ok("🎯 *Model Accuracy*\n\nNo resolved predictions yet.".to_string()),
        }
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

    pub async fn set_bankroll(&self, val: f64) -> Result<()> {
        self.set_f64("bankroll", val).await
    }

    /// Get bankroll for a specific strategy.
    pub async fn strategy_bankroll(&self, strategy: &str) -> Result<f64> {
        self.get_f64(&format!("bankroll:{strategy}")).await
    }

    pub async fn set_strategy_bankroll(&self, strategy: &str, val: f64) -> Result<()> {
        self.set_f64(&format!("bankroll:{strategy}"), val).await
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
    pub async fn init_strategy_bankrolls(
        &self,
        strategies: &[crate::strategy::StrategyProfile],
        strategy_bankroll: f64,
    ) -> Result<()> {
        for s in strategies {
            let key = format!("bankroll:{}", s.name);
            if !self.key_exists(&key).await? {
                self.upsert_f64(&key, strategy_bankroll).await?;
                tracing::info!(
                    strategy = %s.name,
                    bankroll = format_args!("€{strategy_bankroll:.2}"),
                    "Initialized strategy bankroll"
                );
            }
        }

        // Store active strategy names so /stats can show all of them
        let strat_names: Vec<&str> = strategies.iter().map(|s| s.name.as_str()).collect();
        self.upsert_text("active_strategies", &strat_names.join(","))
            .await?;

        // Set starting_bankroll only if not already stored (first run)
        if !self.key_exists("starting_bankroll").await? || self.starting_bankroll().await? == 0.0 {
            let total = strategies.len() as f64 * strategy_bankroll;
            self.upsert_f64("starting_bankroll", total).await?;
            tracing::info!(
                starting = format_args!("€{total:.2}"),
                "Set starting_bankroll (first run)"
            );
        }

        // Init signal counters (idempotent — won't overwrite existing)
        for s in strategies {
            let key = format!("signals_sent_today:{}", s.name);
            if !self.key_exists(&key).await? {
                self.upsert_f64(&key, 0.0).await?;
            }
            let key = format!("last_signal_date:{}", s.name);
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

    pub async fn increment_signals(&self) -> Result<()> {
        let cur = self.get_f64("signals_sent_today").await?;
        self.set_f64("signals_sent_today", cur + 1.0).await
    }

    pub async fn increment_strategy_signals(&self, strategy: &str) -> Result<()> {
        let cur = self
            .get_f64(&format!("signals_sent_today:{strategy}"))
            .await?;
        self.set_f64(&format!("signals_sent_today:{strategy}"), cur + 1.0)
            .await
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

        let row: (i32,) = sqlx::query_as(
            "INSERT INTO bets (market_id, question, side, entry_price, slipped_price, shares, cost, fee_paid, \
             estimated_prob, confidence, edge, kelly_size, reasoning, end_date, context, strategy, source) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17) RETURNING id",
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
        .fetch_one(&self.pool)
        .await?;

        // Deduct from strategy bankroll
        let strat_bankroll = self.strategy_bankroll(&bet.strategy).await?;
        self.set_strategy_bankroll(&bet.strategy, strat_bankroll - bet.cost - bet.fee)
            .await?;
        self.increment_strategy_signals(&bet.strategy).await?;

        // Also update global bankroll
        let bankroll = self.bankroll().await?;
        self.set_bankroll(bankroll - bet.cost - bet.fee).await?;
        self.increment_signals().await?;

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

        sqlx::query(
            "UPDATE bets SET resolved = true, won = $1, pnl = $2, fee_paid = fee_paid + $3, \
             resolved_at = NOW() WHERE id = $4",
        )
        .bind(bet_won)
        .bind(pnl)
        .bind(exit_fee)
        .bind(bet_id)
        .execute(&self.pool)
        .await?;

        // Credit strategy bankroll
        let strat_bankroll = self.strategy_bankroll(&strategy).await?;
        self.set_strategy_bankroll(&strategy, strat_bankroll + net_payout)
            .await?;

        // Also update global bankroll
        let bankroll = self.bankroll().await?;
        self.set_bankroll(bankroll + net_payout).await?;

        // Gather cumulative stats for the message
        let resolved = self.resolved_bets().await?;
        let wins = resolved.iter().filter(|b| b.won == Some(true)).count();
        let losses = resolved.iter().filter(|b| b.won == Some(false)).count();
        let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();

        Ok(Some(ResolvedBet {
            question,
            side,
            won: bet_won,
            entry_price,
            cost,
            shares,
            edge,
            confidence,
            pnl,
            bankroll: bankroll + net_payout,
            total_wins: wins,
            total_losses: losses,
            total_pnl,
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

        let reasoning = format!("Early exit: {reason}");
        sqlx::query(
            "UPDATE bets SET resolved = true, won = false, pnl = $1, fee_paid = fee_paid + $2, \
             reasoning = reasoning || E'\\n' || $3, resolved_at = NOW() WHERE id = $4",
        )
        .bind(pnl)
        .bind(exit_fee)
        .bind(&reasoning)
        .bind(r.id)
        .execute(&self.pool)
        .await?;

        // Credit strategy bankroll with sell proceeds
        let strat_bankroll = self.strategy_bankroll(&r.strategy).await?;
        self.set_strategy_bankroll(&r.strategy, strat_bankroll + net_payout)
            .await?;

        let bankroll = self.bankroll().await?;
        self.set_bankroll(bankroll + net_payout).await?;

        let resolved = self.resolved_bets().await?;
        let wins = resolved.iter().filter(|b| b.won == Some(true)).count();
        let losses = resolved.iter().filter(|b| b.won == Some(false)).count();
        let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();

        Ok(Some(ResolvedBet {
            question: r.question,
            side,
            won: false,
            entry_price: r.entry_price,
            cost: r.cost,
            shares: r.shares,
            edge: r.edge,
            confidence: r.confidence,
            pnl,
            bankroll: bankroll + net_payout,
            total_wins: wins,
            total_losses: losses,
            total_pnl,
            strategy: r.strategy,
            source: r.source,
            market_id: r.market_id,
        }))
    }

    pub async fn open_bet_market_ids(&self) -> Result<Vec<String>> {
        let rows: Vec<(String,)> =
            sqlx::query_as("SELECT market_id FROM bets WHERE resolved = false")
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
        let resolved = self.resolved_bets().await?;
        let open_count = self.open_bets().await?.len();
        let total_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM bets")
            .fetch_one(&self.pool)
            .await?;
        let wins = resolved.iter().filter(|b| b.won == Some(true)).count();
        let losses = resolved.iter().filter(|b| b.won == Some(false)).count();
        let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();
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
        .bind(open_count as i32)
        .bind(total_count.0 as i32)
        .bind(wins as i32)
        .bind(losses as i32)
        .bind(total_pnl)
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
        let resolved = self.resolved_bets().await?;
        let open = self.open_bets().await?;
        let all_bets = self.all_bets().await?;

        let wins = resolved.iter().filter(|b| b.won == Some(true)).count();
        let losses = resolved.iter().filter(|b| b.won == Some(false)).count();
        let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();
        let total_fees: f64 = all_bets.iter().map(|b| b.fee_paid).sum();
        // ROI from actual bankroll delta (most accurate, includes all fees)
        let roi = if starting > 0.0 {
            (bankroll - starting) / starting * 100.0
        } else {
            0.0
        };
        let win_rate = if wins + losses > 0 {
            wins as f64 / (wins + losses) as f64 * 100.0
        } else {
            0.0
        };
        let open_exposure: f64 = open.iter().map(|b| b.cost).sum();

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
            total = all_bets.len(),
            resolved_count = resolved.len(),
            open_count = open.len(),
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

        if !open.is_empty() {
            msg.push_str("\n\u{1f513} *Open bets:*\n");
            for bet in &open {
                msg.push_str(&format!(
                    "\u{2022} _{question}_ \u{2014} {side} \u{20ac}{cost:.2} @ {price:.0}\u{00a2} (edge +{edge:.0}%)\n",
                    question = crate::format::truncate(&bet.question, 50),
                    side = bet.side,
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
        // Build a temporary PortfolioState from DB data for the analysis logic
        let bets = self.all_bets().await?;
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
    placed_at: DateTime<Utc>,
    resolved: bool,
    won: Option<bool>,
    pnl: Option<f64>,
    resolved_at: Option<DateTime<Utc>>,
}

impl BetRow {
    fn into_bet(self) -> Bet {
        let side = if self.side == "No" {
            BetSide::No
        } else {
            BetSide::Yes
        };
        let context: Option<BetContext> = self.context.and_then(|v| serde_json::from_value(v).ok());

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
            placed_at: self.placed_at,
            resolved: self.resolved,
            won: self.won,
            pnl: self.pnl,
            resolved_at: self.resolved_at,
        }
    }
}
