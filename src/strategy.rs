use crate::pricing::kelly::fractional_kelly;
use crate::scanner::live::{Signal, SignalSource};

/// A strategy profile with its own risk parameters and bankroll.
#[derive(Debug, Clone)]
pub struct StrategyProfile {
    pub name: String,
    pub kelly_fraction: f64,
    pub min_effective_edge: f64,
    pub min_confidence: f64,
    pub max_signals_per_day: usize,
    pub min_bet: f64,
}

/// A signal accepted by a strategy, with strategy-specific sizing.
pub struct AcceptedSignal {
    pub kelly_size: f64,
}

impl StrategyProfile {
    pub fn aggressive() -> Self {
        Self {
            name: "aggressive".into(),
            kelly_fraction: 0.50,
            min_effective_edge: 0.05,
            min_confidence: 0.40,
            max_signals_per_day: 10,
            min_bet: 5.0,
        }
    }

    pub fn balanced() -> Self {
        Self {
            name: "balanced".into(),
            kelly_fraction: 0.25,
            min_effective_edge: 0.06,
            min_confidence: 0.40,
            max_signals_per_day: 5,
            min_bet: 5.0,
        }
    }

    pub fn conservative() -> Self {
        Self {
            name: "conservative".into(),
            kelly_fraction: 0.15,
            min_effective_edge: 0.08,
            min_confidence: 0.50,
            max_signals_per_day: 3,
            min_bet: 15.0,
        }
    }

    /// Parse active strategies from comma-separated string.
    /// `max_signals_str` optionally overrides max_signals_per_day per strategy
    /// (format: "aggressive:10,balanced:5,conservative:3").
    pub fn from_config(strategies_str: &str, max_signals_str: &str) -> Vec<Self> {
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

        // Apply per-strategy max signals overrides
        if !max_signals_str.is_empty() {
            for pair in max_signals_str.split(',').map(|s| s.trim()) {
                if let Some((name, val)) = pair.split_once(':') {
                    let name = name.trim().to_lowercase();
                    if let Ok(max) = val.trim().parse::<usize>()
                        && let Some(p) = profiles.iter_mut().find(|p| p.name == name)
                    {
                        p.max_signals_per_day = max;
                    }
                }
            }
        }

        profiles
    }

    /// Evaluate a signal against this strategy's thresholds.
    /// Returns Some with strategy-specific kelly size if accepted, None if rejected.
    /// XGBoost signals get 50% lower thresholds (trust the model).
    pub fn evaluate(&self, signal: &Signal) -> Option<AcceptedSignal> {
        let effective_edge = signal.edge * signal.confidence;

        // XGBoost signals: halve the edge/confidence gates
        let (min_edge, min_conf) = if signal.source == SignalSource::XgBoost {
            (self.min_effective_edge * 0.5, self.min_confidence * 0.7)
        } else {
            (self.min_effective_edge, self.min_confidence)
        };

        if effective_edge < min_edge {
            return None;
        }
        if signal.confidence < min_conf {
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

        // Terminal risk scaling: reduce position size as market approaches expiry.
        // sqrt(days_remaining / 14) — full size at 14d+, scaled down near expiry.
        let terminal_scale = (signal.days_to_expiry / 14.0).clamp(0.1, 1.0).sqrt();
        let scaled_kelly = kelly * terminal_scale;

        Some(AcceptedSignal {
            kelly_size: scaled_kelly,
        })
    }

    pub fn label(&self) -> &str {
        strategy_label(&self.name)
    }
}

/// Get emoji label for a strategy name (usable without a full StrategyProfile).
pub fn strategy_label(name: &str) -> &'static str {
    match name {
        "aggressive" => "🔥",
        "balanced" => "⚖️",
        "conservative" => "🛡️",
        _ => "📊",
    }
}

/// Get emoji icon for a signal source name.
pub fn source_icon(source: &str) -> &'static str {
    match source {
        "xgboost" => "🤖",
        "llm_consensus" => "🧠",
        "copy_trade" => "👥",
        _ => "📊",
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
            polymarket_url: String::new(),
            prior: price,
            combined_lr: 1.0,
            news_matched_count: 0,
            source: SignalSource::LlmConsensus,
            days_to_expiry: 7.0,
            event_slug: None,
            context: BetContext::default(),
            features: None,
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
        let signal = test_signal(0.05, 0.45, 0.50, 0.55);
        // effective_edge = 0.05 * 0.45 = 0.0225 < 0.08 ✗
        assert!(s.evaluate(&signal).is_none());
    }

    #[test]
    fn test_conservative_rejects_low_confidence() {
        let s = StrategyProfile::conservative();
        let signal = test_signal(0.20, 0.35, 0.50, 0.70);
        // effective_edge = 0.07 >= 0.08? no → reject (also conf 0.35 < 0.50 ✗)
        assert!(s.evaluate(&signal).is_none());
    }

    #[test]
    fn test_conservative_accepts_strong_signal() {
        let s = StrategyProfile::conservative();
        let signal = test_signal(0.20, 0.60, 0.50, 0.70);
        // effective_edge = 0.12 >= 0.08 ✓, confidence 0.60 >= 0.50 ✓
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
        let profiles = StrategyProfile::from_config("aggressive, conservative", "");
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].name, "aggressive");
        assert_eq!(profiles[1].name, "conservative");
    }

    #[test]
    fn test_from_config_unknown_fallback() {
        let profiles = StrategyProfile::from_config("invalid", "");
        assert_eq!(profiles.len(), 3); // falls back to all three
    }

    #[test]
    fn test_from_config_single_strategy() {
        let profiles = StrategyProfile::from_config("balanced", "");
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].name, "balanced");
    }

    #[test]
    fn test_from_config_trims_whitespace() {
        let profiles = StrategyProfile::from_config("  Aggressive , BALANCED  ", "");
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].name, "aggressive");
        assert_eq!(profiles[1].name, "balanced");
    }

    #[test]
    fn test_evaluate_no_edge_rejects() {
        // price == prob → zero edge
        let s = StrategyProfile::aggressive();
        let signal = test_signal(0.0, 0.80, 0.50, 0.50);
        assert!(s.evaluate(&signal).is_none());
    }

    #[test]
    fn test_evaluate_kelly_too_small_rejects() {
        // Tiny edge → kelly < 0.01
        let s = StrategyProfile::aggressive();
        let signal = test_signal(0.005, 0.80, 0.50, 0.505);
        assert!(s.evaluate(&signal).is_none());
    }

    #[test]
    fn test_balanced_thresholds() {
        let s = StrategyProfile::balanced();
        // effective_edge = 0.10 * 0.55 = 0.055 < 0.08 → reject
        let weak = test_signal(0.10, 0.55, 0.50, 0.60);
        assert!(s.evaluate(&weak).is_none());

        // effective_edge = 0.20 * 0.55 = 0.11 >= 0.08, conf 0.55 >= 0.50 → accept
        let strong = test_signal(0.20, 0.55, 0.50, 0.70);
        assert!(s.evaluate(&strong).is_some());
    }

    #[test]
    fn test_label() {
        assert_eq!(StrategyProfile::aggressive().label(), "🔥");
        assert_eq!(StrategyProfile::balanced().label(), "⚖️");
        assert_eq!(StrategyProfile::conservative().label(), "🛡️");
    }

    #[test]
    fn test_max_signals_override() {
        let profiles =
            StrategyProfile::from_config("aggressive,balanced", "aggressive:20,balanced:8");
        assert_eq!(profiles[0].max_signals_per_day, 20);
        assert_eq!(profiles[1].max_signals_per_day, 8);
    }

    #[test]
    fn test_max_signals_partial_override() {
        let profiles = StrategyProfile::from_config("aggressive,balanced", "balanced:12");
        assert_eq!(profiles[0].max_signals_per_day, 10); // default
        assert_eq!(profiles[1].max_signals_per_day, 12); // overridden
    }
}
