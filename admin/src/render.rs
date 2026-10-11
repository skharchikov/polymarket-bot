use askama::Template;
use axum::response::Html;

use crate::error::AppError;

pub fn render<T: Template>(t: &T) -> Result<Html<String>, AppError> {
    Ok(Html(t.render()?))
}
