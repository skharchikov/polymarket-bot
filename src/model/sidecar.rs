#![allow(dead_code)]

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::model::features::MarketFeatures;

const MAX_RETRIES: usize = 3;
const RETRY_DELAY: Duration = Duration::from_secs(2);

/// HTTP client for the Python ML sidecar (full stacking ensemble).
#[derive(Debug, Clone)]
pub struct SidecarClient {
    client: Client,
    base_url: String,
}

#[derive(Serialize, Clone)]
struct PredictRequest {
    features: MarketFeatures,
    market_price: f64,
}

#[derive(Serialize, Clone)]
struct BatchRequest {
    items: Vec<PredictRequest>,
}

#[derive(Deserialize)]
pub struct Prediction {
    pub prob: f64,
    pub confidence: f64,
}

#[derive(Deserialize)]
struct BatchResponse {
    predictions: Vec<Prediction>,
}

#[derive(Deserialize)]
pub struct HealthResponse {
    pub model_loaded: bool,
    pub model_age_secs: Option<f64>,
}

impl SidecarClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build sidecar HTTP client");
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Check if the sidecar is up and has a model loaded.
    pub async fn is_healthy(&self) -> bool {
        self.health().await.is_some_and(|h| h.model_loaded)
    }

    /// Get detailed health info from the sidecar.
    pub async fn health(&self) -> Option<HealthResponse> {
        let resp = self
            .client
            .get(format!("{}/health", self.base_url))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.json::<HealthResponse>().await.ok()
    }

    /// Get prediction from the full ensemble, with retries.
    pub async fn predict(
        &self,
        features: &MarketFeatures,
        market_price: f64,
    ) -> Result<Prediction> {
        let req = PredictRequest {
            features: features.clone(),
            market_price,
        };

        let mut last_err = None;
        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                tokio::time::sleep(RETRY_DELAY).await;
            }
            match self.do_predict(&req).await {
                Ok(pred) => return Ok(pred),
                Err(e) => {
                    tracing::debug!(attempt = attempt + 1, err = %e, "Sidecar predict retry");
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap())
    }

    async fn do_predict(&self, req: &PredictRequest) -> Result<Prediction> {
        let resp = self
            .client
            .post(format!("{}/predict", self.base_url))
            .json(req)
            .send()
            .await
            .context("sidecar predict request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("sidecar returned {status}: {body}");
        }

        resp.json::<Prediction>()
            .await
            .context("parsing sidecar response")
    }

    /// Batch prediction with retries.
    pub async fn predict_batch(&self, items: &[(MarketFeatures, f64)]) -> Result<Vec<Prediction>> {
        let req = BatchRequest {
            items: items
                .iter()
                .map(|(features, market_price)| PredictRequest {
                    features: features.clone(),
                    market_price: *market_price,
                })
                .collect(),
        };

        let mut last_err = None;
        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                tokio::time::sleep(RETRY_DELAY).await;
            }
            match self.do_predict_batch(&req).await {
                Ok(preds) => return Ok(preds),
                Err(e) => {
                    tracing::debug!(attempt = attempt + 1, err = %e, "Sidecar batch retry");
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap())
    }

    async fn do_predict_batch(&self, req: &BatchRequest) -> Result<Vec<Prediction>> {
        let resp = self
            .client
            .post(format!("{}/predict_batch", self.base_url))
            .json(req)
            .send()
            .await
            .context("sidecar batch request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("sidecar returned {status}: {body}");
        }

        let batch: BatchResponse = resp.json().await.context("parsing batch response")?;
        Ok(batch.predictions)
    }

    /// Tell the sidecar to reload the model from disk (after retraining).
    pub async fn reload(&self) -> Result<()> {
        self.client
            .post(format!("{}/reload", self.base_url))
            .send()
            .await
            .context("sidecar reload")?;
        Ok(())
    }
}
