use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BetSide {
    Yes,
    No,
}

impl std::fmt::Display for BetSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BetSide::Yes => write!(f, "YES"),
            BetSide::No => write!(f, "NO"),
        }
    }
}

/// Snapshot of market context at bet placement time — for post-mortem learning.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BetContext {
    pub btc_price: f64,
    pub eth_price: f64,
    pub sol_price: f64,
    pub btc_24h_change: f64,
    pub btc_funding_rate: f64,
    pub btc_open_interest: f64,
    pub fear_greed: String,
    pub book_depth: f64,
    pub news_headlines: Vec<String>,
}

/// Parameters for placing a new bet.
pub struct NewBet {
    pub market_id: String,
    pub question: String,
    pub side: BetSide,
    pub entry_price: f64,
    pub slipped_price: f64,
    pub shares: f64,
    pub cost: f64,
    pub fee: f64,
    pub estimated_prob: f64,
    pub confidence: f64,
    pub edge: f64,
    pub kelly_size: f64,
    pub reasoning: String,
    pub end_date: Option<String>,
    pub context: Option<BetContext>,
    pub strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bet {
    pub market_id: String,
    pub question: String,
    #[serde(default = "default_side")]
    pub side: BetSide,
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
    #[serde(default)]
    pub context: Option<BetContext>,
    #[serde(default = "default_strategy")]
    pub strategy: String,
    pub placed_at: DateTime<Utc>,
    pub resolved: bool,
    pub won: Option<bool>,
    pub pnl: Option<f64>,
    pub resolved_at: Option<DateTime<Utc>>,
}

fn default_strategy() -> String {
    "balanced".into()
}

fn default_side() -> BetSide {
    BetSide::Yes
}

/// In-memory portfolio state, used by `PgPortfolio::learning_summary` to
/// generate LLM feedback from the bet history stored in Postgres.
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

impl PortfolioState {
    fn resolved_bets(&self) -> Vec<&Bet> {
        self.bets.iter().filter(|b| b.resolved).collect()
    }

    /// Build a deep analysis of past bets for LLM learning.
    pub fn learning_summary(&self) -> String {
        let resolved = self.resolved_bets();
        if resolved.is_empty() {
            return String::new();
        }

        let wins: Vec<&&Bet> = resolved.iter().filter(|b| b.won == Some(true)).collect();
        let losses: Vec<&&Bet> = resolved.iter().filter(|b| b.won == Some(false)).collect();
        let total = resolved.len();
        let win_rate = if total > 0 {
            wins.len() as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let total_pnl: f64 = resolved.iter().filter_map(|b| b.pnl).sum();

        let mut s = format!(
            "=== LEARNING FROM {total} PAST BETS ===\n\
             Record: {}W/{}L ({win_rate:.0}% win rate), Total PnL: \u{20ac}{total_pnl:+.2}\n\n",
            wins.len(),
            losses.len(),
        );

        // Recent bet details with context
        s.push_str("RECENT BETS (newest first):\n");
        for bet in resolved.iter().rev().take(15) {
            let outcome = match bet.won {
                Some(true) => "WON",
                Some(false) => "LOST",
                None => "PENDING",
            };
            let pnl = bet.pnl.unwrap_or(0.0);
            let duration = bet
                .resolved_at
                .map(|r| {
                    let hours = (r - bet.placed_at).num_hours();
                    if hours < 24 {
                        format!("{hours}h")
                    } else {
                        format!("{}d", hours / 24)
                    }
                })
                .unwrap_or_default();

            s.push_str(&format!(
                "- \"{q}\" | {side} @ {price:.0}c, est {est:.0}%, edge +{edge:.1}%, conf {conf:.0}% | {outcome} \u{20ac}{pnl:+.2} ({duration})\n",
                q = truncate(&bet.question, 55),
                side = bet.side,
                price = bet.entry_price * 100.0,
                est = bet.estimated_prob * 100.0,
                edge = bet.edge * 100.0,
                conf = bet.confidence * 100.0,
            ));
            s.push_str(&format!("  Reasoning: {}\n", truncate(&bet.reasoning, 120)));
            if let Some(ctx) = &bet.context {
                let mut ctx_parts = Vec::new();
                if ctx.btc_price > 0.0 {
                    ctx_parts.push(format!(
                        "BTC ${:.0} ({:+.1}%)",
                        ctx.btc_price, ctx.btc_24h_change
                    ));
                }
                if ctx.btc_funding_rate != 0.0 {
                    ctx_parts.push(format!("funding {:.4}%", ctx.btc_funding_rate * 100.0));
                }
                if !ctx.fear_greed.is_empty() {
                    ctx_parts.push(ctx.fear_greed.clone());
                }
                if ctx.book_depth > 0.0 {
                    ctx_parts.push(format!("depth ${:.0}", ctx.book_depth));
                }
                if !ctx_parts.is_empty() {
                    s.push_str(&format!("  Context: {}\n", ctx_parts.join(", ")));
                }
                if !ctx.news_headlines.is_empty() {
                    s.push_str(&format!(
                        "  News at bet time: {}\n",
                        ctx.news_headlines.join("; ")
                    ));
                }
            }
        }

        // Calibration analysis
        s.push_str("\nCALIBRATION (are our probability estimates accurate?):\n");
        let buckets: &[(f64, f64, &str)] = &[
            (0.0, 0.40, "Low (0-40%)"),
            (0.40, 0.60, "Mid (40-60%)"),
            (0.60, 0.80, "High (60-80%)"),
            (0.80, 1.01, "Very high (80%+)"),
        ];
        for &(lo, hi, label) in buckets {
            let in_bucket: Vec<&&Bet> = resolved
                .iter()
                .filter(|b| b.estimated_prob >= lo && b.estimated_prob < hi)
                .collect();
            if in_bucket.is_empty() {
                continue;
            }
            let avg_est =
                in_bucket.iter().map(|b| b.estimated_prob).sum::<f64>() / in_bucket.len() as f64;
            let actual_rate = in_bucket.iter().filter(|b| b.won == Some(true)).count() as f64
                / in_bucket.len() as f64;
            let gap = actual_rate - avg_est;
            let direction = if gap > 0.05 {
                "UNDERCONFIDENT"
            } else if gap < -0.05 {
                "OVERCONFIDENT"
            } else {
                "well-calibrated"
            };
            s.push_str(&format!(
                "  {label}: {n} bets, avg est {avg:.0}%, actual win {act:.0}% -> {direction}\n",
                n = in_bucket.len(),
                avg = avg_est * 100.0,
                act = actual_rate * 100.0,
            ));
        }

        // Side breakdown
        let yes_bets: Vec<&&Bet> = resolved.iter().filter(|b| b.side == BetSide::Yes).collect();
        let no_bets: Vec<&&Bet> = resolved.iter().filter(|b| b.side == BetSide::No).collect();
        s.push_str("\nSIDE ANALYSIS:\n");
        for (label, bets) in [("YES", &yes_bets), ("NO", &no_bets)] {
            if bets.is_empty() {
                continue;
            }
            let w = bets.iter().filter(|b| b.won == Some(true)).count();
            let pnl: f64 = bets.iter().filter_map(|b| b.pnl).sum();
            s.push_str(&format!(
                "  {label}: {n} bets, {w}W/{l}L ({wr:.0}%), PnL \u{20ac}{pnl:+.2}\n",
                n = bets.len(),
                l = bets.len() - w,
                wr = if bets.is_empty() {
                    0.0
                } else {
                    w as f64 / bets.len() as f64 * 100.0
                },
            ));
        }

        // Confidence accuracy
        s.push_str("\nCONFIDENCE vs REALITY:\n");
        let conf_buckets: &[(f64, f64, &str)] = &[
            (0.0, 0.50, "Low conf (<50%)"),
            (0.50, 0.70, "Medium conf (50-70%)"),
            (0.70, 1.01, "High conf (70%+)"),
        ];
        for &(lo, hi, label) in conf_buckets {
            let in_bucket: Vec<&&Bet> = resolved
                .iter()
                .filter(|b| b.confidence >= lo && b.confidence < hi)
                .collect();
            if in_bucket.is_empty() {
                continue;
            }
            let wr = in_bucket.iter().filter(|b| b.won == Some(true)).count() as f64
                / in_bucket.len() as f64;
            let pnl: f64 = in_bucket.iter().filter_map(|b| b.pnl).sum();
            s.push_str(&format!(
                "  {label}: {n} bets, {wr:.0}% win rate, PnL \u{20ac}{pnl:+.2}\n",
                n = in_bucket.len(),
                wr = wr * 100.0,
            ));
        }

        // Edge size analysis
        s.push_str("\nEDGE SIZE vs OUTCOME:\n");
        let edge_buckets: &[(f64, f64, &str)] = &[
            (0.0, 0.10, "Small edge (<10%)"),
            (0.10, 0.20, "Medium edge (10-20%)"),
            (0.20, 1.0, "Large edge (20%+)"),
        ];
        for &(lo, hi, label) in edge_buckets {
            let in_bucket: Vec<&&Bet> = resolved
                .iter()
                .filter(|b| b.edge >= lo && b.edge < hi)
                .collect();
            if in_bucket.is_empty() {
                continue;
            }
            let wr = in_bucket.iter().filter(|b| b.won == Some(true)).count() as f64
                / in_bucket.len() as f64;
            let pnl: f64 = in_bucket.iter().filter_map(|b| b.pnl).sum();
            s.push_str(&format!(
                "  {label}: {n} bets, {wr:.0}% win rate, PnL \u{20ac}{pnl:+.2}\n",
                n = in_bucket.len(),
                wr = wr * 100.0,
            ));
        }

        // Market context correlation
        let bets_with_ctx: Vec<&&Bet> = resolved.iter().filter(|b| b.context.is_some()).collect();
        if !bets_with_ctx.is_empty() {
            s.push_str("\nCONTEXT PATTERNS:\n");
            let bullish: Vec<&&Bet> = bets_with_ctx
                .iter()
                .filter(|b| b.context.as_ref().unwrap().btc_24h_change > 0.0)
                .copied()
                .collect();
            let bearish: Vec<&&Bet> = bets_with_ctx
                .iter()
                .filter(|b| b.context.as_ref().unwrap().btc_24h_change <= 0.0)
                .copied()
                .collect();
            for (label, bets) in [("BTC bullish day", &bullish), ("BTC bearish day", &bearish)] {
                if bets.is_empty() {
                    continue;
                }
                let wr =
                    bets.iter().filter(|b| b.won == Some(true)).count() as f64 / bets.len() as f64;
                let pnl: f64 = bets.iter().filter_map(|b| b.pnl).sum();
                s.push_str(&format!(
                    "  {label}: {n} bets, {wr:.0}% win rate, PnL \u{20ac}{pnl:+.2}\n",
                    n = bets.len(),
                    wr = wr * 100.0,
                ));
            }

            let pos_funding: Vec<&&Bet> = bets_with_ctx
                .iter()
                .filter(|b| b.context.as_ref().unwrap().btc_funding_rate > 0.0001)
                .copied()
                .collect();
            let neg_funding: Vec<&&Bet> = bets_with_ctx
                .iter()
                .filter(|b| b.context.as_ref().unwrap().btc_funding_rate < -0.0001)
                .copied()
                .collect();
            for (label, bets) in [
                ("Positive funding", &pos_funding),
                ("Negative funding", &neg_funding),
            ] {
                if bets.is_empty() {
                    continue;
                }
                let wr =
                    bets.iter().filter(|b| b.won == Some(true)).count() as f64 / bets.len() as f64;
                s.push_str(&format!(
                    "  {label}: {n} bets, {wr:.0}% win rate\n",
                    n = bets.len(),
                    wr = wr * 100.0,
                ));
            }
        }

        // Actionable lessons
        s.push_str("\nACTIONABLE LESSONS:\n");
        if !losses.is_empty() {
            let avg_lost_edge = losses.iter().map(|b| b.edge).sum::<f64>() / losses.len() as f64;
            let avg_lost_conf =
                losses.iter().map(|b| b.confidence).sum::<f64>() / losses.len() as f64;
            let avg_won_edge = if !wins.is_empty() {
                wins.iter().map(|b| b.edge).sum::<f64>() / wins.len() as f64
            } else {
                0.0
            };
            let avg_won_conf = if !wins.is_empty() {
                wins.iter().map(|b| b.confidence).sum::<f64>() / wins.len() as f64
            } else {
                0.0
            };
            s.push_str(&format!(
                "  Winners avg: edge {:.0}%, conf {:.0}%\n\
                 Losers avg: edge {:.0}%, conf {:.0}%\n",
                avg_won_edge * 100.0,
                avg_won_conf * 100.0,
                avg_lost_edge * 100.0,
                avg_lost_conf * 100.0,
            ));

            if avg_lost_conf > avg_won_conf {
                s.push_str("  WARNING: High confidence correlates with LOSSES. Lower confidence thresholds.\n");
            }
            if avg_lost_edge > avg_won_edge {
                s.push_str("  WARNING: Large perceived edges tend to be WRONG. Be more skeptical of big edges.\n");
            }

            let loss_reasons: Vec<&str> = losses.iter().map(|b| b.reasoning.as_str()).collect();
            let win_reasons: Vec<&str> = wins.iter().map(|b| b.reasoning.as_str()).collect();
            let keywords = [
                "momentum",
                "trend",
                "breakout",
                "support",
                "resistance",
                "bullish",
                "bearish",
                "sentiment",
                "funding",
                "news",
            ];
            for kw in &keywords {
                let in_losses = loss_reasons
                    .iter()
                    .filter(|r| r.to_lowercase().contains(kw))
                    .count();
                let in_wins = win_reasons
                    .iter()
                    .filter(|r| r.to_lowercase().contains(kw))
                    .count();
                let loss_rate = in_losses as f64 / losses.len().max(1) as f64;
                let win_rate_kw = in_wins as f64 / wins.len().max(1) as f64;
                if in_losses + in_wins >= 2 && loss_rate > win_rate_kw + 0.2 {
                    s.push_str(&format!(
                        "  AVOID reasoning based on \"{kw}\" — appears in {:.0}% of losses vs {:.0}% of wins\n",
                        loss_rate * 100.0,
                        win_rate_kw * 100.0,
                    ));
                } else if in_losses + in_wins >= 2 && win_rate_kw > loss_rate + 0.2 {
                    s.push_str(&format!(
                        "  TRUST reasoning based on \"{kw}\" — appears in {:.0}% of wins vs {:.0}% of losses\n",
                        win_rate_kw * 100.0,
                        loss_rate * 100.0,
                    ));
                }
            }
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
