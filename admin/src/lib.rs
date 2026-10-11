pub mod assets;
pub mod auth;
pub mod config;
pub mod db;
pub mod error;
pub mod pages;
pub mod render;
pub mod state;
pub mod view;

use std::sync::Arc;

use axum::{Router, middleware, routing::get};
use tower_http::{catch_panic::CatchPanicLayer, trace::TraceLayer};

use auth::{AuthConfig, require_tailscale_login};
use state::AdminState;

pub fn app(state: AdminState, auth: Arc<AuthConfig>) -> Router {
    let protected = Router::new()
        .route("/", get(pages::home))
        .route("/static/{name}", get(assets::asset))
        .layer(middleware::from_fn_with_state(
            auth,
            require_tailscale_login,
        ))
        .with_state(state);

    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .merge(protected)
        .layer(CatchPanicLayer::new())
        .layer(TraceLayer::new_for_http())
}
