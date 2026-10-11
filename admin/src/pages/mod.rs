use askama::Template;
use axum::response::Html;

use crate::{error::AppError, render::render};

#[derive(Template)]
#[template(path = "pages/placeholder.html")]
struct Placeholder {
    active: &'static str,
}

/// Temporary home page; replaced by the overview in Task 4.
pub async fn home() -> Result<Html<String>, AppError> {
    render(&Placeholder { active: "overview" })
}
