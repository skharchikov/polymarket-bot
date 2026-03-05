use crate::markets::models::Market;
use crate::pricing::kelly::fractional_kelly;
use crate::pricing::probability::ProbabilityEstimate;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub struct StrategyConfig {
    pub edge_threshold: f64,
    pub kelly_fraction: f64,
    pub max_position_pct: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            edge_threshold: 0.05,
            kelly_fraction: 0.25,
            max_position_pct: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Side {
    Yes,
    No,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSignal {
    pub market_id: String,
    pub side: Side,
    pub edge: f64,
    pub position_size_pct: f64,
    pub rationale: String,
}

pub fn evaluate(
    market: &Market,
    estimate: &ProbabilityEstimate,
    bankroll: f64,
    config: &StrategyConfig,
) -> Option<TradeSignal> {
    let edge = estimate.edge(market.implied_probability());

    if edge.abs() < config.edge_threshold {
        return None;
    }

    let (side, price) = if edge > 0.0 {
        (Side::Yes, market.yes_price())
    } else {
        (Side::No, market.no_price())
    };

    let kelly_pct = fractional_kelly(estimate.estimated_probability, price, config.kelly_fraction);
    let position_size_pct = kelly_pct.min(config.max_position_pct);

    if position_size_pct * bankroll < 1.0 {
        return None;
    }

    Some(TradeSignal {
        market_id: market.id.clone(),
        side,
        edge,
        position_size_pct,
        rationale: estimate.rationale.clone(),
    })
}
