use sqlx::PgPool;

#[derive(Clone)]
pub struct AdminState {
    pub db: PgPool,
    pub http: reqwest::Client,
    pub sidecar_url: String,
}
