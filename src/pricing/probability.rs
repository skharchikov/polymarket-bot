use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityEstimate {
    pub market_id: String,
    pub estimated_probability: f64,
    pub confidence: f64,
    pub rationale: String,
}

impl ProbabilityEstimate {
    pub fn edge(&self, market_price: f64) -> f64 {
        self.estimated_probability - market_price
    }

    pub fn has_edge(&self, market_price: f64, threshold: f64) -> bool {
        self.edge(market_price).abs() > threshold
    }
}
