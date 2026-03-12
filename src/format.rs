//! Shared Telegram message formatting utilities.

use crate::storage::portfolio::{Bet, BetSide};
use crate::strategy;

/// Enriched view of an open bet with live market data.
pub struct OpenBetView<'a> {
    pub bet: &'a Bet,
    pub current_yes_price: Option<f64>,
    pub poly_url: Option<String>,
}

impl<'a> OpenBetView<'a> {
    /// Current price on the bet's side.
    fn current_price(&self) -> Option<f64> {
        self.current_yes_price.map(|p| match self.bet.side {
            BetSide::Yes => p,
            BetSide::No => 1.0 - p,
        })
    }

    /// Unrealized PnL in EUR.
    pub fn unrealized(&self) -> Option<f64> {
        self.current_price()
            .map(|cur| self.bet.shares * cur - self.bet.cost)
    }
}

/// Format the open-bets block used in both /open and heartbeat.
///
/// `compact = true`  → one-liner per bet (heartbeat)
/// `compact = false` → full detail per bet (/open command)
pub fn format_open_bets(views: &[OpenBetView], compact: bool) -> String {
    if views.is_empty() {
        return "📭 No open bets".to_string();
    }

    let mut lines = Vec::with_capacity(views.len());
    let mut total_cost = 0.0_f64;
    let mut total_unrealized = 0.0_f64;

    for v in views {
        let bet = v.bet;
        total_cost += bet.cost;

        let strat_label = strategy::strategy_label(&bet.strategy);
        let side_emoji = match bet.side {
            BetSide::Yes => "🟢 YES",
            BetSide::No => "🔴 NO",
        };

        let (pnl_str, _unrealized) = match v.unrealized() {
            Some(u) => {
                total_unrealized += u;
                let pct = if bet.cost > 0.0 {
                    u / bet.cost * 100.0
                } else {
                    0.0
                };
                (format!("€{u:+.2} ({pct:+.0}%)"), u)
            }
            None => ("n/a".to_string(), 0.0),
        };

        if compact {
            let cur_str = match v.current_price() {
                Some(p) => format!("{:.0}¢", p * 100.0),
                None => "?".to_string(),
            };
            let arrow = match v.unrealized() {
                Some(u) if u > 0.01 => "📈",
                Some(u) if u < -0.01 => "📉",
                _ => "➡️",
            };
            lines.push(format!(
                "{arrow} {strat_label} {side} _{q}_ `{entry:.0}¢→{cur}` `{pnl}`",
                side = side_emoji,
                q = truncate(&bet.question, 35),
                entry = bet.entry_price * 100.0,
                cur = cur_str,
                pnl = pnl_str,
            ));
        } else {
            let source_icon = strategy::source_icon(&bet.source);
            let age_days = (chrono::Utc::now() - bet.placed_at).num_hours() as f64 / 24.0;
            let expires = format_expires(bet.end_date.as_deref());

            let cur_pnl = match v.current_price() {
                Some(cur) => format!(
                    " | PnL `{pnl}` @ `{cur:.1}¢`",
                    pnl = pnl_str,
                    cur = cur * 100.0,
                ),
                None => String::new(),
            };

            let q = truncate(&bet.question, 50);
            let q_safe: String = q
                .chars()
                .filter(|c| !matches!(c, '[' | ']' | '(' | ')'))
                .collect();
            let q_link = match &v.poly_url {
                Some(url) if !url.is_empty() => format!("[{q_safe}]({url})"),
                _ => q_safe,
            };

            lines.push(format!(
                "{strat_label} *{side}* `€{cost:.2}` → {shares:.1} shares @ `{price:.1}¢`\n\
                 \u{00a0}\u{00a0}📋 {q_link}\n\
                 \u{00a0}\u{00a0}{source_icon} Edge: `{edge:+.1}%` | Conf: `{conf:.0}%`{cur_pnl}\n\
                 \u{00a0}\u{00a0}⏰ {expires} ({age:.0}d ago)",
                side = side_emoji,
                cost = bet.cost,
                shares = bet.shares,
                price = bet.entry_price * 100.0,
                edge = bet.edge * 100.0,
                conf = bet.confidence * 100.0,
                age = age_days,
            ));
        }
    }

    let unrealized_pct = if total_cost > 0.0 {
        total_unrealized / total_cost * 100.0
    } else {
        0.0
    };

    let header = format!(
        "🔓 *Open Bets* ({count})\n\
         💰 At risk: `€{total_cost:.2}` | Unrealized: `€{total_unrealized:+.2}` ({unrealized_pct:+.1}%)",
        count = views.len(),
    );

    let separator = if compact { "\n" } else { "\n\n" };
    format!("{header}\n\n{}", lines.join(separator))
}

fn format_expires(end_date: Option<&str>) -> String {
    end_date
        .and_then(|d| {
            chrono::DateTime::parse_from_rfc3339(d)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .or_else(|_| {
                    chrono::NaiveDateTime::parse_from_str(d, "%Y-%m-%dT%H:%M:%SZ")
                        .map(|n| n.and_utc())
                })
                .ok()
        })
        .map(|dt| {
            let days_left = (dt - chrono::Utc::now()).num_days();
            format!("{days_left}d left")
        })
        .unwrap_or_default()
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_bet(question: &str, side: BetSide, cost: f64, shares: f64, entry: f64) -> Bet {
        Bet {
            id: 1,
            market_id: "abc123".to_string(),
            question: question.to_string(),
            side,
            entry_price: entry,
            slipped_price: entry,
            shares,
            cost,
            fee_paid: 0.0,
            estimated_prob: 0.6,
            confidence: 0.5,
            edge: 0.1,
            kelly_size: 0.05,
            reasoning: String::new(),
            end_date: None,
            context: None,
            strategy: "aggressive".to_string(),
            source: "xgboost".to_string(),
            url: String::new(),
            event_slug: None,
            placed_at: Utc::now(),
            resolved: false,
            won: None,
            pnl: None,
            resolved_at: None,
        }
    }

    #[test]
    fn test_empty_views() {
        assert_eq!(format_open_bets(&[], false), "📭 No open bets");
        assert_eq!(format_open_bets(&[], true), "📭 No open bets");
    }

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_exact() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_long() {
        assert_eq!(truncate("hello world", 5), "hello...");
    }

    #[test]
    fn test_truncate_unicode() {
        assert_eq!(truncate("héllo wörld", 5), "héllo...");
    }

    #[test]
    fn test_unrealized_yes_profit() {
        let bet = make_bet("Test?", BetSide::Yes, 10.0, 20.0, 0.5);
        let view = OpenBetView {
            bet: &bet,
            current_yes_price: Some(0.6),
            poly_url: None,
        };
        // 20 shares * 0.6 - 10.0 = 2.0
        assert!((view.unrealized().unwrap() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_unrealized_no_bet() {
        let bet = make_bet("Test?", BetSide::No, 10.0, 20.0, 0.5);
        let view = OpenBetView {
            bet: &bet,
            current_yes_price: Some(0.4),
            poly_url: None,
        };
        // NO side price = 1 - 0.4 = 0.6, value = 20 * 0.6 - 10 = 2.0
        assert!((view.unrealized().unwrap() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_unrealized_none_when_no_price() {
        let bet = make_bet("Test?", BetSide::Yes, 10.0, 20.0, 0.5);
        let view = OpenBetView {
            bet: &bet,
            current_yes_price: None,
            poly_url: None,
        };
        assert!(view.unrealized().is_none());
    }

    #[test]
    fn test_compact_format_contains_arrow() {
        let bet = make_bet("Will it rain?", BetSide::Yes, 5.0, 10.0, 0.5);
        let view = OpenBetView {
            bet: &bet,
            current_yes_price: Some(0.6),
            poly_url: None,
        };
        let output = format_open_bets(&[view], true);
        assert!(output.contains("📈")); // price up
        assert!(output.contains("Will it rain?"));
        assert!(output.contains("🔓 *Open Bets* (1)"));
    }

    #[test]
    fn test_full_format_contains_link() {
        let bet = make_bet("Will it rain?", BetSide::Yes, 5.0, 10.0, 0.5);
        let view = OpenBetView {
            bet: &bet,
            current_yes_price: Some(0.6),
            poly_url: Some("https://polymarket.com/event/rain".to_string()),
        };
        let output = format_open_bets(&[view], false);
        assert!(output.contains("[Will it rain?](https://polymarket.com/event/rain)"));
        assert!(output.contains("🤖")); // xgboost source
        assert!(output.contains("🔥")); // aggressive strategy
    }

    #[test]
    fn test_full_format_no_link_when_empty_url() {
        let bet = make_bet("Will it rain?", BetSide::Yes, 5.0, 10.0, 0.5);
        let view = OpenBetView {
            bet: &bet,
            current_yes_price: Some(0.6),
            poly_url: None,
        };
        let output = format_open_bets(&[view], false);
        assert!(output.contains("📋 Will it rain?"));
        assert!(!output.contains("]("));
    }

    #[test]
    fn test_question_special_chars_stripped() {
        let bet = make_bet("Will [Trump] win (2026)?", BetSide::Yes, 5.0, 10.0, 0.5);
        let view = OpenBetView {
            bet: &bet,
            current_yes_price: Some(0.6),
            poly_url: Some("https://example.com".to_string()),
        };
        let output = format_open_bets(&[view], false);
        // Brackets and parens stripped from question in link text
        assert!(output.contains("[Will Trump win 2026?]"));
    }

    #[test]
    fn test_multiple_bets_totals() {
        let bet1 = make_bet("Bet A", BetSide::Yes, 10.0, 20.0, 0.5);
        let bet2 = make_bet("Bet B", BetSide::No, 15.0, 30.0, 0.5);
        let views = vec![
            OpenBetView {
                bet: &bet1,
                current_yes_price: Some(0.6),
                poly_url: None,
            },
            OpenBetView {
                bet: &bet2,
                current_yes_price: Some(0.4),
                poly_url: None,
            },
        ];
        let output = format_open_bets(&views, true);
        assert!(output.contains("(2)"));
        assert!(output.contains("€25.00")); // total cost
    }
}
