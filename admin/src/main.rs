use std::{net::SocketAddr, sync::Arc, time::Duration};

use admin::{auth::AuthConfig, config::AdminConfig, state::AdminState};
use sqlx::postgres::PgPoolOptions;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls CryptoProvider");
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("admin=info".parse()?),
        )
        .init();

    let cfg = AdminConfig::load()?;
    let bind: SocketAddr = cfg.bind.parse()?;
    admin::auth::validate_auth(&cfg.tailscale_login)?;
    admin::auth::validate_bind(&bind, &cfg.dev_bypass())?;

    // Lazy: the admin starts even if the DB (or the admin_ro password) isn't ready;
    // pages show an error until it is.
    let db = PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(3))
        .connect_lazy(&cfg.db_url)?;
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    let prices = Arc::new(admin::prices::PriceCache::new(
        http.clone(),
        Duration::from_secs(60),
    ));
    let state = AdminState {
        db,
        http,
        prices,
        sidecar_url: cfg.sidecar_url.clone(),
    };
    let auth = Arc::new(AuthConfig {
        allowed_login: cfg.tailscale_login.clone(),
        dev_bypass: cfg.dev_bypass(),
    });

    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(%bind, "admin listening");
    axum::serve(listener, admin::app(state, auth)).await?;
    Ok(())
}
