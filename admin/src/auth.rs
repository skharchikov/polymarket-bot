//! Tailscale identity guard. `tailscale serve` injects `Tailscale-User-Login`
//! for tailnet users; the admin only listens on loopback, so the header is the
//! identity.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};

pub const LOGIN_HEADER: &str = "tailscale-user-login";

pub struct AuthConfig {
    pub allowed_login: String,
    /// Dev-only: identity assumed when the header is absent.
    pub dev_bypass: Option<String>,
}

/// Authenticated operator login, inserted into request extensions.
#[derive(Clone, Debug)]
pub struct Operator(pub String);

pub async fn require_tailscale_login(
    State(auth): State<Arc<AuthConfig>>,
    mut req: Request,
    next: Next,
) -> Response {
    let login = req
        .headers()
        .get(LOGIN_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
        .or_else(|| auth.dev_bypass.clone());

    match login {
        Some(l) if l.eq_ignore_ascii_case(&auth.allowed_login) => {
            req.extensions_mut().insert(Operator(l));
            next.run(req).await
        }
        _ => (StatusCode::FORBIDDEN, "forbidden").into_response(),
    }
}

/// Refuse to start with the dev bypass on anything but loopback.
pub fn validate_bind(bind: &SocketAddr, dev_bypass: &Option<String>) -> anyhow::Result<()> {
    if dev_bypass.is_some() && !bind.ip().is_loopback() {
        anyhow::bail!("ADMIN_DEV_BYPASS_LOGIN is set but ADMIN_BIND={bind} is not loopback");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, http::Request, middleware, routing::get};
    use tower::ServiceExt;

    fn guarded(auth: AuthConfig) -> Router {
        Router::new()
            .route("/x", get(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                Arc::new(auth),
                require_tailscale_login,
            ))
    }

    fn cfg(bypass: Option<&str>) -> AuthConfig {
        AuthConfig {
            allowed_login: "owner@example.com".into(),
            dev_bypass: bypass.map(String::from),
        }
    }

    async fn status(app: Router, header: Option<&str>) -> u16 {
        let mut req = Request::builder().uri("/x");
        if let Some(h) = header {
            req = req.header(LOGIN_HEADER, h);
        }
        app.oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
            .as_u16()
    }

    #[tokio::test]
    async fn missing_header_is_forbidden() {
        assert_eq!(status(guarded(cfg(None)), None).await, 403);
    }

    #[tokio::test]
    async fn wrong_login_is_forbidden() {
        assert_eq!(
            status(guarded(cfg(None)), Some("mallory@example.com")).await,
            403
        );
    }

    #[tokio::test]
    async fn matching_login_passes_case_insensitive() {
        assert_eq!(
            status(guarded(cfg(None)), Some("Owner@Example.com")).await,
            200
        );
    }

    #[tokio::test]
    async fn dev_bypass_fills_missing_header() {
        assert_eq!(
            status(guarded(cfg(Some("owner@example.com"))), None).await,
            200
        );
    }

    #[tokio::test]
    async fn dev_bypass_does_not_override_wrong_header() {
        assert_eq!(
            status(
                guarded(cfg(Some("owner@example.com"))),
                Some("mallory@example.com")
            )
            .await,
            403
        );
    }

    #[test]
    fn bypass_on_non_loopback_bind_is_refused() {
        let addr: SocketAddr = "0.0.0.0:8080".parse().unwrap();
        assert!(validate_bind(&addr, &Some("owner@example.com".into())).is_err());
    }

    #[test]
    fn bypass_on_loopback_is_allowed() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        assert!(validate_bind(&addr, &Some("owner@example.com".into())).is_ok());
    }

    #[test]
    fn no_bypass_any_bind_is_allowed() {
        let addr: SocketAddr = "0.0.0.0:8080".parse().unwrap();
        assert!(validate_bind(&addr, &None).is_ok());
    }
}
