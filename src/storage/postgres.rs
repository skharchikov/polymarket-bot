use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::PgPool;

use super::portfolio::{Bet, BetContext, BetSide, NewBet};

/// Returned from resolve_bet with all context needed for a rich Telegram message.
pub struct ResolvedBet {
    pub question: String,
    pub side: BetSide,
    pub won: bool,
    pub entry_price: f64,
    pub cost: f64,
    pub edge: f64,
    pub confidence: f64,
    pub pnl: f64,
    pub bankroll: f64,
    pub total_wins: usize,
    pub total_losses: usize,
    pub total_pnl: f64,
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
        let sql = include_str!("../../migrations/001_init.sql");
        sqlx::raw_sql(sql)
            .execute(&self.pool)
            .await
            .context("failed to run migrations")?;
        Ok(())
    }

    // --- meta helpers ---

    async fn get_f64(&self, key: &str) -> Result<f64> {
        let row: (Option<f64>,) = sqlx::query_as("SELECT value_f64 FROM portfolio WHERE key = $1")
            .bind(key)
            .fetch_one(&self.pool)
            .await?;
        Ok(row.0.unwrap_or(0.0))
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
        let row: (Option<String>,) =
            sqlx::query_as("SELECT value_text FROM portfolio WHERE key = $1")
                .bind(key)
                .fetch_one(&self.pool)
                .await?;
        Ok(row.0.unwrap_or_default())
    }

    async fn set_text(&self, key: &str, val: &str) -> Result<()> {
        sqlx::query("UPDATE portfolio SET value_text = $1 WHERE key = $2")
            .bind(val)
            .bind(key)
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

    pub async fn signals_sent_today(&self) -> Result<usize> {
        Ok(self.get_f64("signals_sent_today").await? as usize)
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

    pub async fn increment_signals(&self) -> Result<()> {
        let cur = self.get_f64("signals_sent_today").await?;
        self.set_f64("signals_sent_today", cur + 1.0).await
    }

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
             estimated_prob, confidence, edge, kelly_size, reasoning, end_date, context) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15) RETURNING id",
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
        .fetch_one(&self.pool)
        .await?;

        let bankroll = self.bankroll().await?;
        self.set_bankroll(bankroll - bet.cost - bet.fee).await?;
        self.increment_signals().await?;

        Ok(row.0)
    }

    pub async fn resolve_bet(&self, market_id: &str, yes_won: bool) -> Result<Option<ResolvedBet>> {
        // Find unresolved bet for this market
        #[derive(sqlx::FromRow)]
        struct OpenBetRow {
            id: i32,
            side: String,
            shares: f64,
            cost: f64,
            question: String,
            entry_price: f64,
            edge: f64,
            confidence: f64,
        }

        let row: Option<OpenBetRow> = sqlx::query_as(
            "SELECT id, side, shares, cost, question, entry_price, edge, confidence \
             FROM bets WHERE market_id = $1 AND resolved = false LIMIT 1",
        )
        .bind(market_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(r) = row else {
            return Ok(None);
        };
        let (bet_id, side_str, shares, cost, question, entry_price, edge, confidence) = (
            r.id,
            r.side,
            r.shares,
            r.cost,
            r.question,
            r.entry_price,
            r.edge,
            r.confidence,
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
        let pnl = net_payout - cost;

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
            edge,
            confidence,
            pnl,
            bankroll: bankroll + net_payout,
            total_wins: wins,
            total_losses: losses,
            total_pnl,
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
        let roi = if starting > 0.0 {
            total_pnl / starting * 100.0
        } else {
            0.0
        };
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let bankroll = self.bankroll().await?;

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
        let roi = if starting > 0.0 {
            total_pnl / starting * 100.0
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

        if !open.is_empty() {
            msg.push_str("\n\u{1f513} *Open bets:*\n");
            for bet in &open {
                msg.push_str(&format!(
                    "\u{2022} _{question}_ \u{2014} {side} \u{20ac}{cost:.2} @ {price:.0}\u{00a2} (edge +{edge:.0}%)\n",
                    question = truncate(&bet.question, 50),
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
    #[allow(dead_code)]
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
            placed_at: self.placed_at,
            resolved: self.resolved,
            won: self.won,
            pnl: self.pnl,
            resolved_at: self.resolved_at,
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
