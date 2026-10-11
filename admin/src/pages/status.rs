use askama::Template;
use axum::{extract::State, response::Html};
use chrono::Utc;

use crate::{
    attention::{COPY_STALE_SECS, Pill, TRADING_STALE_SECS, bot_pill, sidecar_pill},
    db,
    error::AppError,
    render::render,
    sidecar,
    state::AdminState,
};

#[derive(Template)]
#[template(path = "partials/status_pills.html")]
pub struct StatusPills {
    pub pills: Vec<Pill>,
}

pub async fn pills(State(st): State<AdminState>) -> Result<Html<String>, AppError> {
    let now = Utc::now();
    let act = db::status::activity_times(&st.db).await?;
    let model_age = sidecar::health(&st.http, &st.sidecar_url)
        .await
        .ok()
        .and_then(|h| h.model_age_secs);
    render(&StatusPills {
        pills: vec![
            bot_pill("trading-bot", act.trading_last, TRADING_STALE_SECS, now),
            bot_pill("copy-bot", act.copy_last, COPY_STALE_SECS, now),
            sidecar_pill(model_age),
        ],
    })
}
