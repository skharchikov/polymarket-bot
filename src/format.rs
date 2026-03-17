//! Shared Telegram message formatting utilities.

use crate::storage::portfolio::{Bet, BetSide};
use crate::strategy;

/// Format a dollar amount into a compact human-readable string.
///
/// * ≥ 1 000 000 → `$1.2M`
/// * ≥ 1 000     → `$890K`
/// * otherwise   → `$123`
pub fn format_dollars(value: f64) -> String {
    let abs = value.abs();
    let sign = if value < 0.0 { "-" } else { "" };
    if abs >= 1_000_000.0 {
        format!("{sign}${:.1}M", abs / 1_000_000.0)
    } else if abs >= 1_000.0 {
        format!("{sign}${:.0}K", abs / 1_000.0)
    } else {
        format!("{sign}${:.0}", abs)
    }
}

/// Compute win-rate as a percentage.  Returns `0.0` when there are no resolved bets.
pub fn win_rate(wins: usize, losses: usize) -> f64 {
    if wins + losses == 0 {
        return 0.0;
    }
    wins as f64 / (wins + losses) as f64 * 100.0
}

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

/// Per-strategy performance row for `/stats`.
pub struct StratStats {
    pub name: String,
    pub bankroll: f64,
    pub roi: f64,
    pub pnl: f64,
    pub wins: usize,
    pub losses: usize,
    pub open: usize,
}

/// Per-source performance row for `/stats`.
pub struct SourceStats {
    pub name: String,
    pub wins: usize,
    pub losses: usize,
    pub pnl: f64,
}

/// Aggregate copy-trade figures for `/stats`.
pub struct CopyTradeSummary {
    pub traders: usize,
    pub open: usize,
    pub pnl: f64,
}

/// All data required to render the `/stats` message.
pub struct StatsData {
    pub total_bankroll: f64,
    pub starting: f64,
    pub total_pnl: f64,
    pub total_wins: usize,
    pub total_losses: usize,
    pub total_open: usize,
    pub unrealized: f64,
    pub exposure: f64,
    pub strategies: Vec<StratStats>,
    pub sources: Vec<SourceStats>,
    pub copy_trade: Option<CopyTradeSummary>,
}

/// Render the `/stats` message from structured data.
pub fn format_stats(data: &StatsData) -> String {
    let total_roi = if data.starting > 0.0 {
        data.total_pnl / data.starting * 100.0
    } else {
        0.0
    };
    let total_wr = win_rate(data.total_wins, data.total_losses);

    let unrealized_section = if data.total_open > 0 {
        format!(
            "\n📈 Unrealized: `€{:+.2}` (€{:.2} deployed)\n",
            data.unrealized, data.exposure
        )
    } else {
        String::new()
    };

    let strat_lines: Vec<String> = data
        .strategies
        .iter()
        .map(|s| {
            let label = strategy::strategy_label(&s.name);
            let s_wr = win_rate(s.wins, s.losses);
            format!(
                "{label} *{name}*\n\
                 \u{00a0}\u{00a0}💰 `€{bankroll:.2}` | ROI `{roi:+.1}%`\n\
                 \u{00a0}\u{00a0}💵 PnL `€{pnl:+.2}` | {wins}W/{losses}L ({wr:.0}%)\n\
                 \u{00a0}\u{00a0}🔓 {open} open",
                name = s.name,
                bankroll = s.bankroll,
                roi = s.roi,
                pnl = s.pnl,
                wins = s.wins,
                losses = s.losses,
                wr = s_wr,
                open = s.open,
            )
        })
        .collect();

    let mut source_lines = Vec::new();
    for src in &data.sources {
        let label = strategy::source_icon(&src.name);
        let src_wr = win_rate(src.wins, src.losses);
        source_lines.push(format!(
            "{label} *{name}*: {wins}W/{losses}L ({wr:.0}%) | PnL `€{pnl:+.2}`",
            name = src.name,
            wins = src.wins,
            losses = src.losses,
            wr = src_wr,
            pnl = src.pnl,
        ));
    }
    let source_section = if source_lines.is_empty() {
        String::new()
    } else {
        format!("\n\n📡 *By Source*\n{}", source_lines.join("\n"))
    };

    let copy_section = match &data.copy_trade {
        Some(ct) if ct.traders > 0 => format!(
            "\n\n👥 *Copy Trading*: {} traders | {} open | PnL `€{:+.2}`",
            ct.traders, ct.open, ct.pnl
        ),
        _ => String::new(),
    };

    format!(
        "📊 *Bot Statistics*\n\n\
         💰 Bankroll: `€{bankroll:.2}` (started: `€{starting:.2}`)\n\
         💵 Realized PnL: `€{pnl:+.2}` | ROI: `{roi:+.1}%`\n\
         {unrealized_section}\
         📋 {wins}W / {losses}L ({wr:.0}%) | {open} open\n\n\
         {strat_details}{source_section}{copy_section}",
        bankroll = data.total_bankroll,
        starting = data.starting,
        pnl = data.total_pnl,
        roi = total_roi,
        wins = data.total_wins,
        losses = data.total_losses,
        wr = total_wr,
        open = data.total_open,
        strat_details = strat_lines.join("\n\n"),
    )
}

/// A single followed-trader row for `/traders`.
pub struct TraderRow {
    pub name: String,
    pub wallet_short: String,
    pub rank: Option<i32>,
    pub poly_pnl: Option<f64>,
    pub bankroll: f64,
    pub wins: usize,
    pub losses: usize,
    pub pnl: f64,
    pub open: usize,
}

/// Render the `/traders` message from a slice of rows.
pub fn format_traders(traders: &[TraderRow]) -> String {
    if traders.is_empty() {
        return "👥 *Followed Traders*\n\nNo traders followed yet.\nOwner can use `/follow <wallet>` to add one.".to_string();
    }

    let mut lines = vec![format!("👥 *Followed Traders* ({})\n", traders.len())];
    for t in traders {
        let rank = t
            .rank
            .map(|r| format!("#{r}"))
            .unwrap_or_else(|| "—".into());
        let poly_pnl = t
            .poly_pnl
            .map(|p| format!("${:.0}k", p / 1000.0))
            .unwrap_or_else(|| "—".into());
        lines.push(format!(
            "👤 *{name}* (`{short}...`)\n\
             \u{00a0}\u{00a0}🏆 Rank: {rank} | Poly PnL: {poly_pnl}\n\
             \u{00a0}\u{00a0}💰 Bankroll: `€{bankroll:.2}`\n\
             \u{00a0}\u{00a0}📊 Record: {wins}W/{losses}L ({pnl:+.2}€)\n\
             \u{00a0}\u{00a0}🔓 Open: {open}",
            name = t.name,
            short = t.wallet_short,
            bankroll = t.bankroll,
            wins = t.wins,
            losses = t.losses,
            pnl = t.pnl,
            open = t.open,
        ));
    }
    lines.join("\n")
}

/// Data for a copy-trade bet notification.
pub struct CopyBetNotif<'a> {
    pub question: &'a str,
    pub cost: f64,
    pub shares: f64,
    /// Price already multiplied by 100 (cents).
    pub price_cents: f64,
    /// Edge already multiplied by 100 (percent).
    pub edge_pct: f64,
    /// Kelly already multiplied by 100 (percent).
    pub kelly_pct: f64,
    pub ml_info: &'a str,
    pub trader_display: &'a str,
    pub wins: usize,
    pub losses: usize,
    pub trader_pnl: f64,
    pub bankroll: f64,
    pub open: usize,
}

/// Render the copy-trade bet notification message.
pub fn format_copy_bet(n: &CopyBetNotif) -> String {
    format!(
        "👥 *Copy Trade*\n\
         📋 {question}\n\
         💸 *Bet: 🟢 YES*\n\
         💵 Stake: `€{cost:.2}` ({shares:.1} shares @ `{price:.1}¢`)\n\
         📈 Edge: +{edge:.1}% | Kelly: {kelly:.1}%\n\
         {ml_info}\n\
         👤 Trader: `{trader_display}`\n\
         📊 Trader record: {wins}W/{losses}L ({pnl:+.2}€)\n\
         💰 Trader bankroll: `€{bankroll:.2}`\n\
         🔓 Open bets: {open}",
        question = n.question,
        cost = n.cost,
        shares = n.shares,
        price = n.price_cents,
        edge = n.edge_pct,
        kelly = n.kelly_pct,
        ml_info = n.ml_info,
        trader_display = n.trader_display,
        wins = n.wins,
        losses = n.losses,
        pnl = n.trader_pnl,
        bankroll = n.bankroll,
        open = n.open,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_win_rate_zero() {
        assert_eq!(win_rate(0, 0), 0.0);
    }

    #[test]
    fn test_win_rate_all_wins() {
        assert_eq!(win_rate(5, 0), 100.0);
    }

    #[test]
    fn test_win_rate_half() {
        assert!((win_rate(3, 3) - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_format_dollars_thousands() {
        assert_eq!(format_dollars(1500.0), "$2K");
    }

    #[test]
    fn test_format_dollars_small() {
        let s = format_dollars(500.0);
        assert!(s.starts_with('$'));
    }

    #[test]
    fn test_format_dollars_millions() {
        assert_eq!(format_dollars(2_500_000.0), "$2.5M");
    }

    #[test]
    fn test_format_dollars_negative() {
        assert_eq!(format_dollars(-500.0), "-$500");
    }

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

    fn make_stats(starting: f64, total_bankroll: f64, total_pnl: f64) -> StatsData {
        StatsData {
            total_bankroll,
            starting,
            total_pnl,
            total_wins: 0,
            total_losses: 0,
            total_open: 0,
            unrealized: 0.0,
            exposure: 0.0,
            strategies: vec![],
            sources: vec![],
            copy_trade: None,
        }
    }

    // ROI = realized pnl / starting, independent of current bankroll.
    // Before the fix, ROI was (bankroll - starting) / starting which is wrong
    // when money is deployed in open bets (bankroll < starting + pnl).
    #[test]
    fn test_stats_roi_uses_pnl_not_bankroll_diff() {
        // €50 pnl on €1000 start = +5%, regardless of current bankroll
        let data = make_stats(1000.0, 800.0, 50.0); // €200 deployed in open bets
        let output = format_stats(&data);
        assert!(
            output.contains("+5.0%"),
            "expected +5.0% ROI, got: {output}"
        );
    }

    #[test]
    fn test_stats_roi_negative_loss() {
        let data = make_stats(1000.0, 950.0, -100.0);
        let output = format_stats(&data);
        assert!(
            output.contains("-10.0%"),
            "expected -10.0% ROI, got: {output}"
        );
    }

    #[test]
    fn test_stats_roi_zero_when_no_starting() {
        let data = make_stats(0.0, 0.0, 100.0);
        let output = format_stats(&data);
        assert!(
            output.contains("+0.0%"),
            "expected +0.0% ROI, got: {output}"
        );
    }

    #[test]
    fn test_stats_roi_zero_pnl() {
        let data = make_stats(1000.0, 1000.0, 0.0);
        let output = format_stats(&data);
        assert!(
            output.contains("+0.0%"),
            "expected +0.0% ROI, got: {output}"
        );
    }

    #[test]
    fn test_stats_per_strategy_roi_shown() {
        let mut data = make_stats(1000.0, 900.0, 50.0);
        data.strategies = vec![StratStats {
            name: "aggressive".to_string(),
            bankroll: 450.0,
            roi: 10.0, // pre-computed: pnl / starting_per_strat * 100
            pnl: 50.0,
            wins: 3,
            losses: 1,
            open: 2,
        }];
        let output = format_stats(&data);
        assert!(
            output.contains("+10.0%"),
            "expected per-strategy ROI in output: {output}"
        );
        assert!(output.contains("aggressive"));
    }
}
