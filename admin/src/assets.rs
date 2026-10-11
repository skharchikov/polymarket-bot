//! Vendored static assets compiled into the binary.

use axum::{
    extract::Path,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

const ASSETS: &[(&str, &str, &[u8])] = &[
    (
        "htmx.min.js",
        "text/javascript",
        include_bytes!("../static/htmx.min.js"),
    ),
    (
        "uplot.iife.min.js",
        "text/javascript",
        include_bytes!("../static/uplot.iife.min.js"),
    ),
    (
        "uplot.min.css",
        "text/css",
        include_bytes!("../static/uplot.min.css"),
    ),
    ("app.css", "text/css", include_bytes!("../static/app.css")),
    (
        "charts.js",
        "text/javascript",
        include_bytes!("../static/charts.js"),
    ),
];

pub async fn asset(Path(name): Path<String>) -> Response {
    match ASSETS.iter().find(|(n, _, _)| *n == name) {
        Some((_, ctype, bytes)) => (
            [
                (header::CONTENT_TYPE, *ctype),
                (header::CACHE_CONTROL, "public, max-age=3600"),
            ],
            *bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
