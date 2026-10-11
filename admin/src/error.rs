use askama::Template;
use axum::{
    http::StatusCode,
    response::{Html, IntoResponse, Response},
};

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),
    #[error("render error: {0}")]
    Render(#[from] askama::Error),
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

#[derive(Template)]
#[template(path = "partials/error.html")]
struct ErrorPartial<'a> {
    message: &'a str,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        tracing::error!(err = %self, "request failed");
        let msg = self.to_string();
        let body = ErrorPartial { message: &msg }
            .render()
            .unwrap_or_else(|_| "internal error".into());
        (StatusCode::INTERNAL_SERVER_ERROR, Html(body)).into_response()
    }
}
