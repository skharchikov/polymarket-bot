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
    /// Build strategies from config. All parameters come from env vars with defaults.
    pub fn from_config(cfg: &crate::config::AppConfig) -> Vec<Self> {
        let mut profiles = Vec::new();
        for name in cfg.strategies.split(',').map(|s| s.trim().to_lowercase()) {
            let profile = match name.as_str() {
                "aggressive" => Self {
                    name: name.clone(),
                    kelly_fraction: cfg.aggressive_kelly_fraction,
                    min_effective_edge: cfg.aggressive_min_edge,
                    min_confidence: cfg.aggressive_min_confidence,
                    max_signals_per_day: cfg.aggressive_max_signals,
                    min_bet: cfg.aggressive_min_bet,
                },
                "balanced" => Self {
                    name: name.clone(),
                    kelly_fraction: cfg.balanced_kelly_fraction,
                    min_effective_edge: cfg.balanced_min_edge,
                    min_confidence: cfg.balanced_min_confidence,
                    max_signals_per_day: cfg.balanced_max_signals,
                    min_bet: cfg.balanced_min_bet,
                },
                "conservative" => Self {
                    name: name.clone(),
                    kelly_fraction: cfg.conservative_kelly_fraction,
                    min_effective_edge: cfg.conservative_min_edge,
                    min_confidence: cfg.conservative_min_confidence,
                    max_signals_per_day: cfg.conservative_max_signals,
                    min_bet: cfg.conservative_min_bet,
                },
                other => {
                    tracing::warn!(name = other, "Unknown strategy, skipping");
                    continue;
                }
            };
            tracing::info!(
                strategy = %profile.name,
                kelly_fraction = profile.kelly_fraction,
                min_edge = format!("{:.1}%", profile.min_effective_edge * 100.0),
                min_confidence = format!("{:.0}%", profile.min_confidence * 100.0),
                max_signals = profile.max_signals_per_day,
                min_bet = profile.min_bet,
                "Strategy loaded",
            );
            profiles.push(profile);
        }
        if profiles.is_empty() {
            tracing::warn!("No valid strategies configured, defaulting to aggressive");
            profiles.push(Self {
                name: "aggressive".into(),
                kelly_fraction: cfg.aggressive_kelly_fraction,
                min_effective_edge: cfg.aggressive_min_edge,
                min_confidence: cfg.aggressive_min_confidence,
                max_signals_per_day: cfg.aggressive_max_signals,
                min_bet: cfg.aggressive_min_bet,
            });
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

        // Terminal risk scaling: reduce position size for long-dated bets.
        // Data shows >7d bets have 19% WR (-€7.33/bet avg), 1-3d is the sweet spot.
        // Full size at ≤3d, scaled down for longer-dated bets.
        let terminal_scale = if signal.days_to_expiry <= 3.0 {
            1.0
        } else {
            (3.0 / signal.days_to_expiry).sqrt().clamp(0.2, 1.0)
        };
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
            kelly_size: 0.0,
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

    fn make_strategy(
        name: &str,
        kelly: f64,
        edge: f64,
        conf: f64,
        signals: usize,
        min_bet: f64,
    ) -> StrategyProfile {
        StrategyProfile {
            name: name.into(),
            kelly_fraction: kelly,
            min_effective_edge: edge,
            min_confidence: conf,
            max_signals_per_day: signals,
            min_bet,
        }
    }

    fn aggressive() -> StrategyProfile {
        make_strategy("aggressive", 0.50, 0.05, 0.40, 10, 5.0)
    }
    fn balanced() -> StrategyProfile {
        make_strategy("balanced", 0.25, 0.06, 0.40, 5, 5.0)
    }
    fn conservative() -> StrategyProfile {
        make_strategy("conservative", 0.15, 0.08, 0.50, 3, 15.0)
    }

    #[test]
    fn test_aggressive_accepts_low_edge() {
        let s = aggressive();
        let signal = test_signal(0.10, 0.55, 0.50, 0.60);
        // effective_edge = 0.10 * 0.55 = 0.055 >= 0.05 ✓
        assert!(s.evaluate(&signal).is_some());
    }

    #[test]
    fn test_conservative_rejects_low_edge() {
        let s = conservative();
        let signal = test_signal(0.05, 0.45, 0.50, 0.55);
        // effective_edge = 0.05 * 0.45 = 0.0225 < 0.08 ✗
        assert!(s.evaluate(&signal).is_none());
    }

    #[test]
    fn test_conservative_rejects_low_confidence() {
        let s = conservative();
        let signal = test_signal(0.20, 0.35, 0.50, 0.70);
        // effective_edge = 0.07 >= 0.08? no → reject (also conf 0.35 < 0.50 ✗)
        assert!(s.evaluate(&signal).is_none());
    }

    #[test]
    fn test_conservative_accepts_strong_signal() {
        let s = conservative();
        let signal = test_signal(0.20, 0.60, 0.50, 0.70);
        // effective_edge = 0.12 >= 0.08 ✓, confidence 0.60 >= 0.50 ✓
        assert!(s.evaluate(&signal).is_some());
    }

    #[test]
    fn test_aggressive_kelly_larger_than_conservative() {
        let signal = test_signal(0.20, 0.80, 0.50, 0.70);
        let agg = aggressive().evaluate(&signal).unwrap();
        let con = conservative().evaluate(&signal).unwrap();
        assert!(
            agg.kelly_size > con.kelly_size,
            "aggressive kelly {} should be > conservative {}",
            agg.kelly_size,
            con.kelly_size,
        );
    }

    #[test]
    fn test_from_config_parses() {
        let mut cfg = crate::config::AppConfig::test_default();
        cfg.strategies = "aggressive, conservative".into();
        let profiles = StrategyProfile::from_config(&cfg);
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].name, "aggressive");
        assert_eq!(profiles[1].name, "conservative");
    }

    #[test]
    fn test_from_config_unknown_fallback() {
        let mut cfg = crate::config::AppConfig::test_default();
        cfg.strategies = "invalid".into();
        let profiles = StrategyProfile::from_config(&cfg);
        assert_eq!(profiles.len(), 1); // falls back to aggressive
        assert_eq!(profiles[0].name, "aggressive");
    }

    #[test]
    fn test_from_config_single_strategy() {
        let mut cfg = crate::config::AppConfig::test_default();
        cfg.strategies = "balanced".into();
        let profiles = StrategyProfile::from_config(&cfg);
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].name, "balanced");
    }

    #[test]
    fn test_from_config_trims_whitespace() {
        let mut cfg = crate::config::AppConfig::test_default();
        cfg.strategies = "  Aggressive , BALANCED  ".into();
        let profiles = StrategyProfile::from_config(&cfg);
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].name, "aggressive");
        assert_eq!(profiles[1].name, "balanced");
    }

    #[test]
    fn test_evaluate_no_edge_rejects() {
        let s = aggressive();
        let signal = test_signal(0.0, 0.80, 0.50, 0.50);
        assert!(s.evaluate(&signal).is_none());
    }

    #[test]
    fn test_evaluate_kelly_too_small_rejects() {
        let s = aggressive();
        let signal = test_signal(0.005, 0.80, 0.50, 0.505);
        assert!(s.evaluate(&signal).is_none());
    }

    #[test]
    fn test_balanced_thresholds() {
        let s = balanced();
        // effective_edge = 0.10 * 0.55 = 0.055 < 0.06 → reject
        let weak = test_signal(0.10, 0.55, 0.50, 0.60);
        assert!(s.evaluate(&weak).is_none());

        // effective_edge = 0.20 * 0.55 = 0.11 >= 0.06, conf 0.55 >= 0.40 → accept
        let strong = test_signal(0.20, 0.55, 0.50, 0.70);
        assert!(s.evaluate(&strong).is_some());
    }

    #[test]
    fn test_label() {
        assert_eq!(aggressive().label(), "🔥");
        assert_eq!(balanced().label(), "⚖️");
        assert_eq!(conservative().label(), "🛡️");
    }

    #[test]
    fn test_env_override_applies() {
        let mut cfg = crate::config::AppConfig::test_default();
        cfg.strategies = "aggressive".into();
        cfg.aggressive_min_edge = 0.02;
        cfg.aggressive_max_signals = 30;
        let profiles = StrategyProfile::from_config(&cfg);
        assert_eq!(profiles[0].min_effective_edge, 0.02);
        assert_eq!(profiles[0].max_signals_per_day, 30);
    }
}
