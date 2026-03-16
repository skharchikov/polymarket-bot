#![allow(dead_code)]

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Pure-Rust XGBoost inference from exported JSON model.
/// Traverses decision trees without any native XGBoost dependency.
#[derive(Debug)]
pub struct XgbModel {
    trees: Vec<Tree>,
    base_score: f64,
    scaler: Option<Scaler>,
}

#[derive(Debug)]
struct Tree {
    nodes: Vec<Node>,
}

#[derive(Debug)]
enum Node {
    Split {
        feature_idx: usize,
        threshold: f64,
        yes: usize,
        no: usize,
        missing: usize,
    },
    Leaf {
        value: f64,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct Scaler {
    pub center: Vec<f64>,
    pub scale: Vec<f64>,
    pub feature_names: Vec<String>,
}

impl Scaler {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("reading scaler from {}", path.display()))?;
        serde_json::from_str(&data).context("parsing scaler JSON")
    }

    pub fn transform(&self, features: &[f64]) -> Vec<f64> {
        features
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let center = self.center.get(i).copied().unwrap_or(0.0);
                let scale = self.scale.get(i).copied().unwrap_or(1.0);
                if scale == 0.0 {
                    0.0
                } else {
                    (v - center) / scale
                }
            })
            .collect()
    }
}

impl XgbModel {
    /// Load model from XGBoost's JSON export format.
    /// Optionally loads a companion scaler file (same path with .scaler.json suffix).
    pub fn load(model_path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(model_path)
            .with_context(|| format!("reading model from {}", model_path.display()))?;
        let raw: RawModel = serde_json::from_str(&data).context("parsing XGBoost JSON")?;

        let base_score = raw
            .learner
            .learner_model_param
            .base_score
            .parse::<f64>()
            .unwrap_or(0.5);

        let mut trees = Vec::new();
        for raw_tree in &raw.learner.gradient_booster.model.trees {
            trees.push(parse_tree(raw_tree)?);
        }

        // Try loading scaler
        let scaler_path = model_path.with_extension("scaler.json");
        let scaler = if scaler_path.exists() {
            Some(Scaler::load(&scaler_path)?)
        } else {
            None
        };

        tracing::info!(
            n_trees = trees.len(),
            base_score,
            has_scaler = scaler.is_some(),
            "Loaded XGBoost model"
        );

        Ok(Self {
            trees,
            base_score,
            scaler,
        })
    }

    /// Predict probability of YES outcome.
    pub fn predict_prob(&self, features: &[f64]) -> f64 {
        let scaled = match &self.scaler {
            Some(s) => s.transform(features),
            None => features.to_vec(),
        };

        let raw: f64 = self.trees.iter().map(|t| t.predict(&scaled)).sum();
        sigmoid(raw + logit(self.base_score))
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}

impl Tree {
    fn predict(&self, features: &[f64]) -> f64 {
        let mut node_idx = 0;
        loop {
            match &self.nodes[node_idx] {
                Node::Leaf { value } => return *value,
                Node::Split {
                    feature_idx,
                    threshold,
                    yes,
                    no,
                    missing,
                } => {
                    let val = features.get(*feature_idx).copied();
                    node_idx = match val {
                        None => *missing,
                        Some(v) if v.is_nan() => *missing,
                        Some(v) if v < *threshold => *yes,
                        Some(_) => *no,
                    };
                }
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    (p / (1.0 - p)).ln()
}

// ---- JSON parsing for XGBoost's native export format ----

#[derive(Deserialize)]
struct RawModel {
    learner: RawLearner,
}

#[derive(Deserialize)]
struct RawLearner {
    learner_model_param: RawModelParam,
    gradient_booster: RawBooster,
}

#[derive(Deserialize)]
struct RawModelParam {
    base_score: String,
}

#[derive(Deserialize)]
struct RawBooster {
    model: RawGBTree,
}

#[derive(Deserialize)]
struct RawGBTree {
    trees: Vec<RawTree>,
}

#[derive(Deserialize)]
struct RawTree {
    split_indices: Vec<usize>,
    split_conditions: Vec<f64>,
    left_children: Vec<i64>,
    right_children: Vec<i64>,
    default_left: Vec<u8>,
}

fn parse_tree(raw: &RawTree) -> Result<Tree> {
    let n = raw.split_indices.len();
    let mut nodes = Vec::with_capacity(n);

    for i in 0..n {
        let left = raw.left_children[i];
        let right = raw.right_children[i];

        if left == -1 && right == -1 {
            // Leaf node — split_conditions holds the leaf value
            nodes.push(Node::Leaf {
                value: raw.split_conditions[i],
            });
        } else {
            let yes = left as usize;
            let no = right as usize;
            let missing = if raw.default_left.get(i).copied().unwrap_or(0) == 1 {
                yes
            } else {
                no
            };
            nodes.push(Node::Split {
                feature_idx: raw.split_indices[i],
                threshold: raw.split_conditions[i],
                yes,
                no,
                missing,
            });
        }
    }

    Ok(Tree { nodes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_logit_roundtrip() {
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let roundtrip = sigmoid(logit(p));
            assert!(
                (roundtrip - p).abs() < 1e-10,
                "logit/sigmoid roundtrip failed for {p}"
            );
        }
    }

    #[test]
    fn test_scaler_transform() {
        let scaler = Scaler {
            center: vec![10.0, 20.0],
            scale: vec![2.0, 5.0],
            feature_names: vec!["a".into(), "b".into()],
        };
        let result = scaler.transform(&[12.0, 30.0]);
        assert!((result[0] - 1.0).abs() < 1e-10); // (12-10)/2 = 1
        assert!((result[1] - 2.0).abs() < 1e-10); // (30-20)/5 = 2
    }

    #[test]
    fn test_scaler_zero_scale() {
        let scaler = Scaler {
            center: vec![5.0],
            scale: vec![0.0],
            feature_names: vec!["a".into()],
        };
        let result = scaler.transform(&[10.0]);
        assert!((result[0]).abs() < 1e-10); // Should return 0 for zero scale
    }

    #[test]
    fn test_load_real_model() {
        let model_path = std::path::Path::new("model/xgb_model.json");
        if !model_path.exists() {
            eprintln!(
                "Skipping: model/xgb_model.json not found (run scripts/train_model.py first)"
            );
            return;
        }
        let model = XgbModel::load(model_path).expect("Failed to load model");
        assert!(model.n_trees() > 0, "Model should have trees");

        // Test prediction with a typical market feature vector matching MarketFeatures::NAMES:
        // [yes_price, momentum_1h, momentum_24h, volatility_24h, rsi,
        //  log_volume, days_to_expiry, is_crypto,
        //  price_change_1d, price_change_1w, days_since_created, created_to_expiry_span]
        let features = vec![
            0.55, 0.02, -0.05, 0.03, 0.6, 12.0, 15.0, 1.0, 0.03, -0.02, 20.0, 30.0,
        ];
        let prob = model.predict_prob(&features);
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability must be in [0,1]: {prob}"
        );

        println!("Model: {} trees, prob={prob:.4}", model.n_trees());
    }
}
