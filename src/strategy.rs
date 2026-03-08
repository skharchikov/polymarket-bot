use crate::pricing::kelly::fractional_kelly;
use crate::scanner::live::Signal;

/// A strategy profile with its own risk parameters and bankroll.
#[derive(Debug, Clone)]
pub struct StrategyProfile {
    pub name: String,
    pub kelly_fraction: f64,
    pub min_effective_edge: f64,
    pub min_confidence: f64,
    pub max_signals_per_day: usize,
    pub min_bet: f64,
    /// Fraction of total bankroll allocated to this strategy at init.
    pub bankroll_share: f64,
}

/// A signal accepted by a strategy, with strategy-specific sizing.
pub struct AcceptedSignal {
    pub strategy: String,
    pub kelly_size: f64,
}

impl StrategyProfile {
    pub fn aggressive() -> Self {
        Self {
            name: "aggressive".into(),
            kelly_fraction: 0.50,
            min_effective_edge: 0.05,
            min_confidence: 0.40,
            max_signals_per_day: 5,
            min_bet: 5.0,
            bankroll_share: 0.30,
        }
    }

    pub fn balanced() -> Self {
        Self {
            name: "balanced".into(),
            kelly_fraction: 0.25,
            min_effective_edge: 0.08,
            min_confidence: 0.50,
            max_signals_per_day: 3,
            min_bet: 10.0,
            bankroll_share: 0.40,
        }
    }

    pub fn conservative() -> Self {
        Self {
            name: "conservative".into(),
            kelly_fraction: 0.10,
            min_effective_edge: 0.12,
            min_confidence: 0.65,
            max_signals_per_day: 2,
            min_bet: 15.0,
            bankroll_share: 0.30,
        }
    }

    /// Parse active strategies from comma-separated string.
    pub fn from_config(strategies_str: &str) -> Vec<Self> {
        let mut profiles = Vec::new();
        for name in strategies_str.split(',').map(|s| s.trim().to_lowercase()) {
            match name.as_str() {
                "aggressive" => profiles.push(Self::aggressive()),
                "balanced" => profiles.push(Self::balanced()),
                "conservative" => profiles.push(Self::conservative()),
                other => {
                    tracing::warn!(name = other, "Unknown strategy, skipping");
                }
            }
        }
        if profiles.is_empty() {
            tracing::warn!("No valid strategies configured, defaulting to all three");
            profiles = vec![Self::aggressive(), Self::balanced(), Self::conservative()];
        }

        // Normalize bankroll shares to sum to 1.0
        let total_share: f64 = profiles.iter().map(|p| p.bankroll_share).sum();
        if total_share > 0.0 {
            for p in &mut profiles {
                p.bankroll_share /= total_share;
            }
        }

        profiles
    }

    /// Evaluate a signal against this strategy's thresholds.
    /// Returns Some with strategy-specific kelly size if accepted, None if rejected.
    pub fn evaluate(&self, signal: &Signal) -> Option<AcceptedSignal> {
        let effective_edge = signal.edge * signal.confidence;
        if effective_edge < self.min_effective_edge {
            return None;
        }
        if signal.confidence < self.min_confidence {
            return None;
        }

        let kelly = fractional_kelly(
            signal.estimated_prob,
            signal.current_price,
            self.kelly_fraction,
        );
        if kelly < 0.01 {
            return None;
        }

        Some(AcceptedSignal {
            strategy: self.name.clone(),
            kelly_size: kelly,
        })
    }

    pub fn label(&self) -> &str {
        match self.name.as_str() {
            "aggressive" => "🔥",
            "balanced" => "⚖️",
            "conservative" => "🛡️",
            _ => "📊",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::portfolio::{BetContext, BetSide};

    fn test_signal(edge: f64, confidence: f64, price: f64, prob: f64) -> Signal {
        Signal {
            market_id: "test".into(),
            question: "Test?".into(),
            side: BetSide::Yes,
            current_price: price,
            estimated_prob: prob,
            confidence,
            edge,
            kelly_size: 0.0, // unused by evaluate()
            reasoning: "test".into(),
            end_date: None,
            volume: 1000.0,
            context: BetContext::default(),
        }
    }

    #[test]
    fn test_aggressive_accepts_low_edge() {
        let s = StrategyProfile::aggressive();
        let signal = test_signal(0.10, 0.55, 0.50, 0.60);
        // effective_edge = 0.10 * 0.55 = 0.055 >= 0.05 ✓
        assert!(s.evaluate(&signal).is_some());
    }

    #[test]
    fn test_conservative_rejects_low_edge() {
        let s = StrategyProfile::conservative();
        let signal = test_signal(0.10, 0.55, 0.50, 0.60);
        // effective_edge = 0.055 < 0.12 ✗
        assert!(s.evaluate(&signal).is_none());
    }

    #[test]
    fn test_conservative_rejects_low_confidence() {
        let s = StrategyProfile::conservative();
        let signal = test_signal(0.30, 0.60, 0.50, 0.80);
        // effective_edge = 0.18 >= 0.12 ✓, but confidence 0.60 < 0.65 ✗
        assert!(s.evaluate(&signal).is_none());
    }

    #[test]
    fn test_conservative_accepts_strong_signal() {
        let s = StrategyProfile::conservative();
        let signal = test_signal(0.20, 0.80, 0.50, 0.70);
        // effective_edge = 0.16 >= 0.12 ✓, confidence 0.80 >= 0.65 ✓
        assert!(s.evaluate(&signal).is_some());
    }

    #[test]
    fn test_aggressive_kelly_larger_than_conservative() {
        let signal = test_signal(0.20, 0.80, 0.50, 0.70);
        let agg = StrategyProfile::aggressive().evaluate(&signal).unwrap();
        let con = StrategyProfile::conservative().evaluate(&signal).unwrap();
        assert!(
            agg.kelly_size > con.kelly_size,
            "aggressive kelly {} should be > conservative {}",
            agg.kelly_size,
            con.kelly_size,
        );
    }

    #[test]
    fn test_from_config_parses() {
        let profiles = StrategyProfile::from_config("aggressive, conservative");
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].name, "aggressive");
        assert_eq!(profiles[1].name, "conservative");
        // Shares should be normalized
        let total: f64 = profiles.iter().map(|p| p.bankroll_share).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_from_config_unknown_fallback() {
        let profiles = StrategyProfile::from_config("invalid");
        assert_eq!(profiles.len(), 3); // falls back to all three
    }
}
