#![allow(dead_code)]

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// HTTP client for the Python ML sidecar (full stacking ensemble).
#[derive(Debug, Clone)]
pub struct SidecarClient {
    client: Client,
    base_url: String,
}

#[derive(Serialize)]
struct PredictRequest {
    features: Vec<f64>,
    market_price: f64,
}

#[derive(Serialize)]
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
struct HealthResponse {
    status: String,
    model_loaded: bool,
}

impl SidecarClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("failed to build sidecar HTTP client");
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Check if the sidecar is up and has a model loaded.
    pub async fn is_healthy(&self) -> bool {
        match self
            .client
            .get(format!("{}/health", self.base_url))
            .send()
            .await
        {
            Ok(resp) => resp
                .json::<HealthResponse>()
                .await
                .map(|h| h.model_loaded)
                .unwrap_or(false),
            Err(_) => false,
        }
    }

    /// Get prediction from the full ensemble for a single market.
    pub async fn predict(&self, features: &[f64], market_price: f64) -> Result<Prediction> {
        let req = PredictRequest {
            features: features.to_vec(),
            market_price,
        };
        let resp = self
            .client
            .post(format!("{}/predict", self.base_url))
            .json(&req)
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

    /// Batch prediction for multiple markets at once.
    pub async fn predict_batch(&self, items: &[(Vec<f64>, f64)]) -> Result<Vec<Prediction>> {
        let req = BatchRequest {
            items: items
                .iter()
                .map(|(features, market_price)| PredictRequest {
                    features: features.clone(),
                    market_price: *market_price,
                })
                .collect(),
        };
        let resp = self
            .client
            .post(format!("{}/predict_batch", self.base_url))
            .json(&req)
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
