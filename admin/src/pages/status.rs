use askama::Template;
use axum::{extract::State, response::Html};
use chrono::{DateTime, Utc};

use crate::{
    attention::{COPY_STALE_SECS, Pill, PillState, TRADING_STALE_SECS, bot_pill, sidecar_pill},
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

/// Build the pill row; a failed activity query degrades both bot pills to Down.
pub fn build_pills(
    act: Result<db::status::ActivityTimes, String>,
    model_age: Option<f64>,
    now: DateTime<Utc>,
) -> Vec<Pill> {
    let mut pills = match act {
        Ok(a) => vec![
            bot_pill("trading-bot", a.trading_last, TRADING_STALE_SECS, now),
            bot_pill("copy-bot", a.copy_last, COPY_STALE_SECS, now),
        ],
        Err(_) => ["trading-bot", "copy-bot"]
            .into_iter()
            .map(|name| Pill {
                name,
                state: PillState::Down,
                detail: "db error".into(),
            })
            .collect(),
    };
    pills.push(sidecar_pill(model_age));
    pills
}

pub async fn pills(State(st): State<AdminState>) -> Result<Html<String>, AppError> {
    let now = Utc::now();
    let act = db::status::activity_times(&st.db).await.map_err(|e| {
        tracing::error!(err = %e, "status pills: activity query failed");
        e.to_string()
    });
    let model_age = sidecar::health(&st.http, &st.sidecar_url)
        .await
        .ok()
        .and_then(|h| h.model_age_secs);
    render(&StatusPills {
        pills: build_pills(act, model_age, now),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_error_degrades_bot_pills_but_keeps_sidecar() {
        let p = build_pills(Err("boom".into()), Some(10.0), Utc::now());
        assert_eq!(p.len(), 3);
        assert!(matches!(p[0].state, PillState::Down) && p[0].detail == "db error");
        assert!(matches!(p[1].state, PillState::Down) && p[1].detail == "db error");
        assert_eq!(p[2].name, "sidecar");
    }
}
