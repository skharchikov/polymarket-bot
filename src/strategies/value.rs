use crate::pricing::kelly::fractional_kelly;

use super::signal::{Side, Signal, SignalSource};

pub struct ValueStrategy {
    pub edge_threshold: f64,
    pub kelly_fraction: f64,
    pub max_size_pct: f64,
}

impl Default for ValueStrategy {
    fn default() -> Self {
        Self {
            edge_threshold: 0.05,
            kelly_fraction: 0.25,
            max_size_pct: 0.05,
        }
    }
}

impl ValueStrategy {
    pub fn evaluate(
        &self,
        market_id: &str,
        market_price: f64,
        model_prob: f64,
        confidence: f64,
    ) -> Option<Signal> {
        let edge = model_prob - market_price;
        if edge.abs() < self.edge_threshold {
            return None;
        }

        // Scale edge by confidence — low confidence reduces effective edge
        let effective_edge = edge * confidence;
        if effective_edge.abs() < self.edge_threshold * 0.5 {
            return None;
        }

        let (side, price) = if edge > 0.0 {
            (Side::Yes, market_price)
        } else {
            (Side::No, 1.0 - market_price)
        };

        let prob_for_kelly = if edge > 0.0 {
            model_prob
        } else {
            1.0 - model_prob
        };

        let size =
            fractional_kelly(prob_for_kelly, price, self.kelly_fraction).min(self.max_size_pct);

        if size < 0.001 {
            return None;
        }

        Some(Signal {
            market_id: market_id.to_string(),
            side,
            edge,
            confidence,
            size_pct: size,
            source: SignalSource::Value {
                model_prob,
                market_prob: market_price,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_edge() {
        let s = ValueStrategy::default();
        assert!(s.evaluate("m1", 0.60, 0.60, 1.0).is_none());
    }

    #[test]
    fn test_buy_yes_on_positive_edge() {
        let s = ValueStrategy::default();
        let sig = s.evaluate("m1", 0.50, 0.70, 0.9).unwrap();
        assert_eq!(sig.side, Side::Yes);
        assert!(sig.edge > 0.0);
        assert!(sig.size_pct > 0.0);
    }

    #[test]
    fn test_buy_no_on_negative_edge() {
        let s = ValueStrategy::default();
        let sig = s.evaluate("m1", 0.80, 0.30, 0.9).unwrap();
        assert_eq!(sig.side, Side::No);
        assert!(sig.edge < 0.0);
    }

    #[test]
    fn test_low_confidence_filters() {
        let s = ValueStrategy::default();
        // Edge is 0.06 but confidence 0.3 → effective edge 0.018 < 0.025 threshold
        assert!(s.evaluate("m1", 0.50, 0.56, 0.3).is_none());
    }
}
