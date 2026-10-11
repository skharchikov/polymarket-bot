use askama::Template;
use axum::{extract::State, response::Html};

use crate::{error::AppError, render::render, sidecar, state::AdminState, view::age};

#[derive(Template)]
#[template(path = "pages/model.html")]
pub struct ModelPage {
    pub active: &'static str,
    pub health: Result<Vec<(String, String)>, String>,
    pub model_age: String,
    pub retrain: Result<Vec<(String, String)>, String>,
}

pub async fn page(State(st): State<AdminState>) -> Result<Html<String>, AppError> {
    let (h, r) = tokio::join!(
        sidecar::get_json(&st.http, &st.sidecar_url, "/health"),
        sidecar::get_json(&st.http, &st.sidecar_url, "/retrain/status"),
    );
    let model_age = h
        .as_ref()
        .ok()
        .and_then(|v| v.get("model_age_secs"))
        .and_then(|v| v.as_f64())
        .map(age)
        .unwrap_or_else(|| "—".into());
    render(&ModelPage {
        active: "model",
        health: h.map(|v| sidecar::flatten(&v)).map_err(|e| e.to_string()),
        model_age,
        retrain: r.map(|v| sidecar::flatten(&v)).map_err(|e| e.to_string()),
    })
}
