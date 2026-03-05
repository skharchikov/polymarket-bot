use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

const STATE_FILE: &str = "portfolio_state.json";

/// Slippage: price moves against us by this fraction when entering.
const SLIPPAGE_PCT: f64 = 0.01;
/// Polymarket trading fee (2% on profit, approximated as 2% of notional).
const FEE_PCT: f64 = 0.02;
/// Minimum bet size in EUR.
const MIN_BET: f64 = 10.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bet {
    pub market_id: String,
    pub question: String,
    pub entry_price: f64,
    pub slipped_price: f64,
    pub shares: f64,
    pub cost: f64,
    pub fee_paid: f64,
    pub estimated_prob: f64,
    pub confidence: f64,
    pub edge: f64,
    pub kelly_size: f64,
    pub reasoning: String,
    pub end_date: Option<String>,
    pub placed_at: DateTime<Utc>,
    pub resolved: bool,
    pub won: Option<bool>,
    pub pnl: Option<f64>,
    pub resolved_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailySnapshot {
    pub date: String,
    pub bankroll: f64,
    pub open_bets: usize,
    pub total_bets: usize,
    pub wins: usize,
    pub losses: usize,
    pub total_pnl: f64,
    pub roi_pct: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PortfolioState {
    pub starting_bankroll: f64,
    pub bankroll: f64,
    pub bets: Vec<Bet>,
    pub daily_snapshots: Vec<DailySnapshot>,
    pub signals_sent_today: usize,
    pub last_signal_date: String,
    pub last_daily_report_date: String,
}

impl PortfolioState {
    pub fn new(starting_bankroll: f64) -> Self {
        Self {
            starting_bankroll,
            bankroll: starting_bankroll,
            bets: Vec::new(),
            daily_snapshots: Vec::new(),
            signals_sent_today: 0,
            last_signal_date: String::new(),
            last_daily_report_date: String::new(),
        }
    }

    pub fn load_or_create(starting_bankroll: f64) -> Self {
        let path = Path::new(STATE_FILE);
        if path.exists() {
            match std::fs::read_to_string(path) {
                Ok(content) => match serde_json::from_str(&content) {
                    Ok(state) => {
                        tracing::info!(bankroll = %format_args!("€{:.2}", starting_bankroll), "Loaded portfolio state");
                        return state;
                    }
                    Err(e) => {
                        tracing::warn!(err = %e, "Failed to parse state file, starting fresh");
                    }
                },
                Err(e) => {
                    tracing::warn!(err = %e, "Failed to read state file, starting fresh");
                }
            }
        }
        tracing::info!(bankroll = %format_args!("€{:.2}", starting_bankroll), "Starting fresh portfolio");
        Self::new(starting_bankroll)
    }

    pub fn save(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(STATE_FILE, json).context("failed to write portfolio state")?;
        Ok(())
    }

    /// Reset daily signal counter if it's a new day.
    pub fn reset_daily_if_needed(&mut self) {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        if self.last_signal_date != today {
            self.signals_sent_today = 0;
            self.last_signal_date = today;
        }
    }

    /// Place a paper bet with realistic slippage and fees.
    /// Returns the bet details or None if we can't afford it.
    pub fn place_bet(
        &mut self,
        market_id: &str,
        question: &str,
        entry_price: f64,
        kelly_size: f64,
        estimated_prob: f64,
        confidence: f64,
        edge: f64,
        reasoning: &str,
        end_date: Option<&str>,
    ) -> Option<Bet> {
        // Calculate bet size: kelly fraction of bankroll, but at least MIN_BET
        let raw_bet = self.bankroll * kelly_size;
        let bet_amount = raw_bet.max(MIN_BET);

        if bet_amount > self.bankroll {
            tracing::warn!(
                bankroll = format_args!("€{:.2}", self.bankroll),
                bet = format_args!("€{:.2}", bet_amount),
                "Insufficient bankroll"
            );
            return None;
        }

        // Slippage: we pay more than the displayed price
        let slipped_price = (entry_price * (1.0 + SLIPPAGE_PCT)).min(0.99);

        // Shares = bet_amount / slipped_price
        let shares = bet_amount / slipped_price;

        // Fee on the trade
        let fee = bet_amount * FEE_PCT;

        // Total cost = bet + fee
        let total_cost = bet_amount + fee;

        if total_cost > self.bankroll {
            tracing::warn!(
                bankroll = format_args!("€{:.2}", self.bankroll),
                cost = format_args!("€{:.2}", total_cost),
                "Insufficient bankroll after fees"
            );
            return None;
        }

        self.bankroll -= total_cost;

        let bet = Bet {
            market_id: market_id.to_string(),
            question: question.to_string(),
            entry_price,
            slipped_price,
            shares,
            cost: bet_amount,
            fee_paid: fee,
            estimated_prob,
            confidence,
            edge,
            kelly_size,
            reasoning: reasoning.to_string(),
            end_date: end_date.map(|s| s.to_string()),
            placed_at: Utc::now(),
            resolved: false,
            won: None,
            pnl: None,
            resolved_at: None,
        };

        self.bets.push(bet.clone());
        self.signals_sent_today += 1;
        let _ = self.save();

        Some(bet)
    }

    /// Resolve a bet. If YES won, payout = shares * 1.0 minus exit fee.
    /// If NO won, payout = 0.
    pub fn resolve_bet(&mut self, market_id: &str, yes_won: bool) -> Option<f64> {
        let bet = self
            .bets
            .iter_mut()
            .find(|b| b.market_id == market_id && !b.resolved)?;

        bet.resolved = true;
        bet.won = Some(yes_won);
        bet.resolved_at = Some(Utc::now());

        let gross_payout = if yes_won { bet.shares } else { 0.0 };
        let exit_fee = gross_payout * FEE_PCT;
        let net_payout = gross_payout - exit_fee;
        let pnl = net_payout - bet.cost;

        bet.pnl = Some(pnl);
        bet.fee_paid += exit_fee;

        self.bankroll += net_payout;
        let _ = self.save();

        Some(pnl)
    }

    /// Check if we already have an open bet on this market.
    pub fn has_open_bet(&self, market_id: &str) -> bool {
        self.bets
            .iter()
            .any(|b| b.market_id == market_id && !b.resolved)
    }

    pub fn open_bets(&self) -> Vec<&Bet> {
        self.bets.iter().filter(|b| !b.resolved).collect()
    }

    pub fn resolved_bets(&self) -> Vec<&Bet> {
        self.bets.iter().filter(|b| b.resolved).collect()
    }

    /// Generate daily stats summary.
    pub fn daily_summary(&self) -> String {
        let resolved = self.resolved_bets();
        let open = self.open_bets();
        let wins = resolved.iter().filter(|b| b.won == Some(true)).count();
        let losses = resolved.iter().filter(|b| b.won == Some(false)).count();
        let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();
        let total_fees: f64 = self.bets.iter().map(|b| b.fee_paid).sum();
        let roi = if self.starting_bankroll > 0.0 {
            total_pnl / self.starting_bankroll * 100.0
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
            "📊 *Daily Portfolio Report*\n\n\
             💰 Bankroll: `€{bankroll:.2}`\n\
             📈 Starting: `€{start:.2}`\n\
             💵 Total PnL: `€{pnl:+.2}`\n\
             📉 Total fees: `€{fees:.2}`\n\
             🎯 ROI: `{roi:+.1}%`\n\n\
             📋 *Stats:*\n\
             Total bets: {total}\n\
             Resolved: {resolved_count} (✅ {wins}W / ❌ {losses}L)\n\
             Win rate: {win_rate:.0}%\n\
             Open: {open_count} (€{open_exposure:.2} exposed)\n",
            bankroll = self.bankroll,
            start = self.starting_bankroll,
            pnl = total_pnl,
            fees = total_fees,
            roi = roi,
            total = self.bets.len(),
            resolved_count = resolved.len(),
            wins = wins,
            losses = losses,
            win_rate = win_rate,
            open_count = open.len(),
            open_exposure = open_exposure,
        );

        if !open.is_empty() {
            msg.push_str("\n🔓 *Open bets:*\n");
            for bet in &open {
                msg.push_str(&format!(
                    "• _{question}_ — €{cost:.2} @ {price:.0}¢ (edge +{edge:.0}%)\n",
                    question = truncate(&bet.question, 50),
                    cost = bet.cost,
                    price = bet.slipped_price * 100.0,
                    edge = bet.edge * 100.0,
                ));
            }
        }

        msg
    }

    /// Take a daily snapshot.
    pub fn take_snapshot(&mut self) {
        let resolved = self.resolved_bets();
        let wins = resolved.iter().filter(|b| b.won == Some(true)).count();
        let losses = resolved.iter().filter(|b| b.won == Some(false)).count();
        let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();
        let roi = total_pnl / self.starting_bankroll * 100.0;

        let snapshot = DailySnapshot {
            date: Utc::now().format("%Y-%m-%d").to_string(),
            bankroll: self.bankroll,
            open_bets: self.open_bets().len(),
            total_bets: self.bets.len(),
            wins,
            losses,
            total_pnl,
            roi_pct: roi,
        };

        self.daily_snapshots.push(snapshot);
        let _ = self.save();
    }

    /// Check if we should send a daily report.
    pub fn should_send_daily_report(&self) -> bool {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        self.last_daily_report_date != today
    }

    pub fn mark_daily_report_sent(&mut self) {
        self.last_daily_report_date = Utc::now().format("%Y-%m-%d").to_string();
        let _ = self.save();
    }

    /// Build a summary of past resolved bets for LLM learning.
    /// Shows what we bet on, our estimate vs reality, and whether we won/lost.
    pub fn learning_summary(&self) -> String {
        let resolved = self.resolved_bets();
        if resolved.is_empty() {
            return String::new();
        }

        // Take last 10 resolved bets (most recent mistakes/wins)
        let recent: Vec<&&Bet> = resolved.iter().rev().take(10).collect();

        let wins = resolved.iter().filter(|b| b.won == Some(true)).count();
        let losses = resolved.len() - wins;

        let mut s = format!(
            "Overall record: {wins}W/{losses}L ({:.0}% win rate)\n",
            if wins + losses > 0 {
                wins as f64 / (wins + losses) as f64 * 100.0
            } else {
                0.0
            }
        );

        for bet in &recent {
            let outcome = match bet.won {
                Some(true) => "WON",
                Some(false) => "LOST",
                None => "PENDING",
            };
            let pnl = bet.pnl.unwrap_or(0.0);
            s.push_str(&format!(
                "- \"{question}\" | bought YES @ {price:.0}¢, est {est:.0}%, edge +{edge:.0}% | {outcome} (€{pnl:+.2}) | reason: {reason}\n",
                question = truncate(&bet.question, 60),
                price = bet.entry_price * 100.0,
                est = bet.estimated_prob * 100.0,
                edge = bet.edge * 100.0,
                reason = truncate(&bet.reasoning, 80),
            ));
        }

        // Add pattern analysis
        let lost_bets: Vec<&&Bet> = resolved.iter().filter(|b| b.won == Some(false)).collect();
        if !lost_bets.is_empty() {
            let avg_lost_edge: f64 =
                lost_bets.iter().map(|b| b.edge).sum::<f64>() / lost_bets.len() as f64;
            let avg_lost_conf: f64 =
                lost_bets.iter().map(|b| b.confidence).sum::<f64>() / lost_bets.len() as f64;
            s.push_str(&format!(
                "PATTERN: Lost bets had avg edge {:.0}%, avg confidence {:.0}%. Be more cautious with similar profiles.\n",
                avg_lost_edge * 100.0,
                avg_lost_conf * 100.0,
            ));
        }

        s
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
