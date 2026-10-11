use askama::Template;
use axum::{extract::State, response::Html};
use chrono::Utc;

use crate::{
    db::{self, DEFAULT_MODE},
    error::AppError,
    render::render,
    state::AdminState,
    view::{ago, money, pnl_class, short_wallet, signed_money},
};

pub struct TraderView {
    pub name: String,
    pub profile: String,
    pub rank: String,
    pub poly_all: String,
    pub poly_month: String,
    pub bets: String,
    pub our_pnl: String,
    pub class: &'static str,
    pub seen: String,
    pub active: bool,
}

#[derive(Template)]
#[template(path = "pages/traders.html")]
pub struct TradersPage {
    pub active: &'static str,
    pub rows: Vec<TraderView>,
}

pub async fn page(State(st): State<AdminState>) -> Result<Html<String>, AppError> {
    let now = Utc::now();
    let rows = db::traders::traders(&st.db, DEFAULT_MODE)
        .await?
        .into_iter()
        .map(|t| TraderView {
            name: t
                .username
                .clone()
                .unwrap_or_else(|| short_wallet(&t.proxy_wallet)),
            profile: format!("https://polymarket.com/profile/{}", t.proxy_wallet),
            rank: t
                .rank
                .map(|r| format!("#{r}"))
                .unwrap_or_else(|| "—".into()),
            poly_all: t.pnl.map(money).unwrap_or_else(|| "—".into()),
            poly_month: t.pnl_month.map(signed_money).unwrap_or_else(|| "—".into()),
            bets: format!("{} ({} open)", t.our_bets, t.our_open),
            our_pnl: signed_money(t.our_pnl),
            class: pnl_class(t.our_pnl),
            seen: t
                .last_checked_at
                .map(|s| ago((now - s).num_seconds() as f64))
                .unwrap_or_else(|| "never".into()),
            active: t.active,
        })
        .collect();
    render(&TradersPage {
        active: "traders",
        rows,
    })
}
