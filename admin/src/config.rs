use confique::Config;

#[derive(Debug, Config)]
pub struct AdminConfig {
    /// Postgres URL for the read-only `admin_ro` role.
    #[config(env = "ADMIN_DB_URL")]
    pub db_url: String,

    /// Listen address. Loopback outside docker.
    #[config(env = "ADMIN_BIND", default = "127.0.0.1:8080")]
    pub bind: String,

    /// The only Tailscale login allowed in.
    #[config(env = "ADMIN_TAILSCALE_LOGIN")]
    pub tailscale_login: String,

    /// Dev only: login assumed when the Tailscale header is absent.
    #[config(env = "ADMIN_DEV_BYPASS_LOGIN", default = "")]
    pub dev_bypass_login: String,

    /// Model sidecar base URL.
    #[config(env = "MODEL_SIDECAR_URL", default = "http://model-server:8000")]
    pub sidecar_url: String,
}

impl AdminConfig {
    pub fn load() -> Result<Self, confique::Error> {
        Self::builder().env().load()
    }

    pub fn dev_bypass(&self) -> Option<String> {
        (!self.dev_bypass_login.is_empty()).then(|| self.dev_bypass_login.clone())
    }
}
