use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Health {
    pub status: String,
    pub model_loaded: bool,
    pub model_age_secs: Option<f64>,
}

pub async fn health(http: &reqwest::Client, base: &str) -> anyhow::Result<Health> {
    Ok(http
        .get(format!("{}/health", base.trim_end_matches('/')))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

pub async fn get_json(
    http: &reqwest::Client,
    base: &str,
    path: &str,
) -> anyhow::Result<serde_json::Value> {
    Ok(http
        .get(format!("{}{path}", base.trim_end_matches('/')))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}
