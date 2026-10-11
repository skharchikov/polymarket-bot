use askama::Template;
use axum::{extract::State, response::Html};

use crate::{
    db::{
        self,
        signals::{Blocked, ReasonCount, SkipCount},
    },
    error::AppError,
    render::render,
    state::AdminState,
    view::{money, short_wallet},
};

fn opt(v: Option<f64>, scale: f64, decimals: usize) -> String {
    v.map(|x| format!("{:.*}", decimals, x * scale))
        .unwrap_or_else(|| "—".into())
}

pub struct NearView {
    pub time: String,
    pub question: String,
    pub reason: String,
    pub price: String,
    pub prob: String,
    pub edge: String,
    pub conf: String,
}

pub struct EventView {
    pub time: String,
    pub who: String,
    pub market_id: String,
    pub side: String,
    pub price: String,
    pub size: String,
    pub copied: bool,
    pub skip: String,
}

#[derive(Template)]
#[template(path = "pages/signals.html")]
pub struct SignalsPage {
    pub active: &'static str,
    pub reasons: Vec<ReasonCount>,
    pub near: Vec<NearView>,
    pub blocked: Vec<Blocked>,
    pub events: Vec<EventView>,
    pub skips: Vec<SkipCount>,
}

pub async fn page(State(st): State<AdminState>) -> Result<Html<String>, AppError> {
    let (reasons, near, blocked, events, skips) = tokio::try_join!(
        db::signals::reasons(&st.db),
        db::signals::near_misses(&st.db, 20),
        db::signals::correlation_blocked(&st.db, 20),
        db::signals::copy_events(&st.db, 50),
        db::signals::copy_skip_counts(&st.db),
    )?;
    let near = near
        .into_iter()
        .map(|n| NearView {
            time: n.created_at.format("%H:%M").to_string(),
            question: n.question,
            reason: n.reason,
            price: opt(n.current_price, 1.0, 2),
            prob: opt(n.estimated_prob, 1.0, 2),
            edge: opt(n.edge, 100.0, 1),
            conf: opt(n.confidence, 1.0, 2),
        })
        .collect();
    let events = events
        .into_iter()
        .map(|e| EventView {
            time: e.detected_at.format("%m-%d %H:%M").to_string(),
            who: e
                .username
                .clone()
                .unwrap_or_else(|| short_wallet(&e.trader_wallet)),
            market_id: e.market_id,
            side: e.side,
            price: format!("{:.2}", e.price),
            size: money(e.size_usd),
            copied: e.acted_on,
            skip: e.skip_reason.unwrap_or_else(|| "—".into()),
        })
        .collect();
    render(&SignalsPage {
        active: "signals",
        reasons,
        near,
        blocked,
        events,
        skips,
    })
}
