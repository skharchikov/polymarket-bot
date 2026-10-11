use askama::Template;
use axum::{
    extract::{Query, State},
    http::header,
    response::{Html, IntoResponse, Response},
};
use chrono::NaiveDate;
use serde::Deserialize;

use super::nonempty;
use crate::{
    csv,
    db::{self, DEFAULT_MODE, history::HistoryFilter, positions::Filter},
    error::AppError,
    render::render,
    state::AdminState,
    view::{money, pnl_class, signed_money, signed_pct},
};

const PAGE_SIZE: i64 = 50;
const CSV_MAX_ROWS: i64 = 100_000;

#[derive(Debug, Default, Deserialize)]
pub struct HistoryQuery {
    pub bot: Option<String>,
    pub source: Option<String>,
    pub strategy: Option<String>,
    pub outcome: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    pub page: Option<i64>,
}

impl HistoryQuery {
    fn filter(&self) -> HistoryFilter {
        let date = |s: &Option<String>| {
            nonempty(s.clone()).and_then(|d| NaiveDate::parse_from_str(&d, "%Y-%m-%d").ok())
        };
        HistoryFilter {
            base: Filter {
                bot: nonempty(self.bot.clone()),
                source: nonempty(self.source.clone()),
                strategy: nonempty(self.strategy.clone()),
            },
            outcome: nonempty(self.outcome.clone()),
            from: date(&self.from),
            to: date(&self.to),
        }
    }

    /// Query string without `page`, for pager links and the CSV link.
    fn qs(&self) -> String {
        let pairs = [
            ("bot", &self.bot),
            ("source", &self.source),
            ("strategy", &self.strategy),
            ("outcome", &self.outcome),
            ("from", &self.from),
            ("to", &self.to),
        ];
        pairs
            .iter()
            .filter_map(|(k, v)| nonempty((*v).clone()).map(|v| format!("{k}={}", urlencode(&v))))
            .collect::<Vec<_>>()
            .join("&")
    }
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

pub struct RowView {
    pub resolved: String,
    pub bot: String,
    pub strategy: String,
    pub question: String,
    pub url: String,
    pub side: String,
    pub cost: String,
    pub outcome: String,
    pub pnl: String,
    pub class: &'static str,
}

pub struct BreakdownView {
    pub bot: String,
    pub source: String,
    pub strategy: String,
    pub category: String,
    pub n: i64,
    pub win_rate: String,
    pub pnl: String,
    pub roi: String,
    pub class: &'static str,
}

#[derive(Template)]
#[template(path = "pages/history.html")]
pub struct HistoryPage {
    pub active: &'static str,
    pub rows: Vec<RowView>,
    pub breakdown: Vec<BreakdownView>,
    pub total: i64,
    pub page: i64,
    pub pages: i64,
    pub qs: String,
    pub q: HistoryQuery,
}

pub async fn page(
    State(st): State<AdminState>,
    Query(q): Query<HistoryQuery>,
) -> Result<Html<String>, AppError> {
    let f = q.filter();
    let page = q.page.unwrap_or(1).max(1);
    let ((rows, total), bd) = tokio::try_join!(
        db::history::resolved(&st.db, DEFAULT_MODE, &f, PAGE_SIZE, (page - 1) * PAGE_SIZE),
        db::history::breakdown(&st.db, DEFAULT_MODE, &f),
    )?;
    let rows = rows
        .into_iter()
        .map(|r| {
            let pnl = r.pnl.unwrap_or(0.0);
            RowView {
                resolved: r
                    .resolved_at
                    .map(|t| t.format("%Y-%m-%d").to_string())
                    .unwrap_or_default(),
                bot: r.bot,
                strategy: r.strategy,
                question: r.question,
                // Only link out to https URLs; anything else renders as plain text.
                url: if r.url.starts_with("https://") {
                    r.url
                } else {
                    String::new()
                },
                side: r.side.to_uppercase(),
                cost: money(r.cost),
                outcome: match r.won {
                    Some(true) => "won",
                    Some(false) => "lost",
                    None => "—",
                }
                .into(),
                pnl: signed_money(pnl),
                class: pnl_class(pnl),
            }
        })
        .collect();
    let breakdown = bd
        .into_iter()
        .map(|b| BreakdownView {
            win_rate: if b.n > 0 {
                format!("{:.0}%", b.wins as f64 / b.n as f64 * 100.0)
            } else {
                "—".into()
            },
            pnl: signed_money(b.pnl),
            roi: signed_pct(b.roi),
            class: pnl_class(b.pnl),
            bot: b.bot,
            source: b.source,
            strategy: b.strategy,
            category: b.category,
            n: b.n,
        })
        .collect();
    render(&HistoryPage {
        active: "history",
        rows,
        breakdown,
        total,
        page,
        pages: ((total + PAGE_SIZE - 1) / PAGE_SIZE).max(1),
        qs: q.qs(),
        q,
    })
}

pub async fn csv_export(
    State(st): State<AdminState>,
    Query(q): Query<HistoryQuery>,
) -> Result<Response, AppError> {
    let (rows, _) =
        db::history::resolved(&st.db, DEFAULT_MODE, &q.filter(), CSV_MAX_ROWS, 0).await?;
    Ok((
        [
            (header::CONTENT_TYPE, "text/csv; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"bets.csv\"",
            ),
        ],
        csv::write_rows(&rows),
    )
        .into_response())
}
