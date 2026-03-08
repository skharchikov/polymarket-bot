//! Bayesian updating for prediction market signals.
//!
//! Instead of averaging raw probabilities from LLM agents, we:
//! 1. Start with market price as prior (it encodes all public info)
//! 2. Each agent estimates a likelihood ratio (LR) — how much more likely
//!    is this news in YES-worlds vs NO-worlds
//! 3. Update sequentially: posterior_odds = prior_odds × LR₁ × LR₂ × ...
//! 4. Convert back to probability
//!
//! This is more principled than averaging because:
//! - LRs compose multiplicatively (independent evidence stacks correctly)
//! - Anchoring to market price prevents hallucinated probabilities
//! - Each agent's contribution is auditable

/// Convert probability to odds: p / (1 - p)
pub fn prob_to_odds(p: f64) -> f64 {
    let p = p.clamp(0.001, 0.999);
    p / (1.0 - p)
}

/// Convert odds to probability: odds / (1 + odds)
pub fn odds_to_prob(odds: f64) -> f64 {
    if odds <= 0.0 {
        return 0.001;
    }
    (odds / (1.0 + odds)).clamp(0.001, 0.999)
}

/// Dampen a likelihood ratio by confidence.
///
/// When confidence is 1.0, the LR is used as-is.
/// When confidence is 0.0, the LR becomes 1.0 (no update).
/// This prevents low-confidence agents from moving the posterior.
///
/// Uses log-space interpolation: LR_dampened = LR^confidence
pub fn dampen_lr(lr: f64, confidence: f64) -> f64 {
    let confidence = confidence.clamp(0.0, 1.0);
    if lr <= 0.0 {
        return 1.0;
    }
    lr.powf(confidence)
}

/// An assessment from a single agent: likelihood ratio + confidence + reasoning.
#[derive(Debug, Clone)]
pub struct AgentAssessment {
    pub role: String,
    /// How much more likely is this news in YES-worlds vs NO-worlds.
    /// LR > 1.0 means evidence favors YES; LR < 1.0 means evidence favors NO.
    /// LR = 1.0 means news is uninformative.
    pub likelihood_ratio: f64,
    /// 0.0 = no useful info, 1.0 = highly confident in the LR estimate.
    pub confidence: f64,
    pub reasoning: String,
}

/// Result of Bayesian updating across multiple agents.
#[derive(Debug, Clone)]
pub struct BayesianEstimate {
    #[allow(dead_code)]
    pub prior: f64,
    pub posterior: f64,
    pub combined_lr: f64,
    pub confidence: f64,
    pub reasoning: String,
}

/// Run Bayesian update: start from market price, apply each agent's
/// confidence-dampened likelihood ratio sequentially.
pub fn bayesian_update(prior: f64, assessments: &[AgentAssessment]) -> BayesianEstimate {
    if assessments.is_empty() {
        return BayesianEstimate {
            prior,
            posterior: prior,
            combined_lr: 1.0,
            confidence: 0.0,
            reasoning: "No assessments".to_string(),
        };
    }

    let prior_odds = prob_to_odds(prior);
    let mut combined_lr = 1.0;

    for a in assessments {
        let dampened = dampen_lr(a.likelihood_ratio, a.confidence);
        combined_lr *= dampened;
    }

    let posterior_odds = prior_odds * combined_lr;
    let posterior = odds_to_prob(posterior_odds);

    // Confidence: geometric mean of agent confidences, penalized by disagreement.
    // Disagreement = agents pulling in opposite directions (some LR>1, some LR<1).
    let conf_product: f64 = assessments.iter().map(|a| a.confidence).product();
    let geo_mean_conf = conf_product.powf(1.0 / assessments.len() as f64);

    let agreement = compute_agreement(assessments);
    let confidence = (geo_mean_conf * agreement).clamp(0.0, 1.0);

    // Build reasoning string
    let mut parts: Vec<String> = assessments
        .iter()
        .map(|a| {
            format!(
                "[{} LR={:.2} @{:.0}%] {}",
                a.role,
                a.likelihood_ratio,
                a.confidence * 100.0,
                a.reasoning,
            )
        })
        .collect();
    parts.push(format!(
        "Bayes: {:.1}% -> {:.1}% (LR={:.2}, agreement={:.0}%)",
        prior * 100.0,
        posterior * 100.0,
        combined_lr,
        agreement * 100.0,
    ));

    BayesianEstimate {
        prior,
        posterior,
        combined_lr,
        confidence,
        reasoning: parts.join(" | "),
    }
}

/// Measure how much agents agree on direction.
/// Returns 1.0 when all LRs point the same way, drops toward 0.0 when they conflict.
fn compute_agreement(assessments: &[AgentAssessment]) -> f64 {
    if assessments.len() <= 1 {
        return 1.0;
    }

    // Convert LRs to log-space (positive = favors YES, negative = favors NO)
    let log_lrs: Vec<f64> = assessments
        .iter()
        .map(|a| a.likelihood_ratio.max(0.01).ln())
        .collect();

    let mean_log = log_lrs.iter().sum::<f64>() / log_lrs.len() as f64;
    let variance =
        log_lrs.iter().map(|l| (l - mean_log).powi(2)).sum::<f64>() / log_lrs.len() as f64;
    let std_dev = variance.sqrt();

    // Agreement decays as log-LR spread increases.
    // std_dev of ~0.5 (LRs differ by ~1.6x) gives agreement ~0.6
    // std_dev of ~1.0 (LRs differ by ~2.7x) gives agreement ~0.37
    (-std_dev).exp()
}

/// Compute the edge and optimal side from a Bayesian estimate.
/// Returns (side_is_yes, edge, bet_price, bet_prob).
pub fn compute_edge(
    estimate: &BayesianEstimate,
    current_yes_price: f64,
) -> Option<(bool, f64, f64, f64)> {
    let yes_edge = estimate.posterior - current_yes_price;
    let no_price = 1.0 - current_yes_price;
    let no_prob = 1.0 - estimate.posterior;
    let no_edge = no_prob - no_price;

    if yes_edge >= no_edge && yes_edge > 0.0 {
        Some((true, yes_edge, current_yes_price, estimate.posterior))
    } else if no_edge > 0.0 {
        Some((false, no_edge, no_price, no_prob))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- prob_to_odds / odds_to_prob ---

    #[test]
    fn test_prob_odds_roundtrip() {
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let roundtrip = odds_to_prob(prob_to_odds(p));
            assert!((roundtrip - p).abs() < 1e-9, "roundtrip failed for {p}");
        }
    }

    #[test]
    fn test_prob_to_odds_even() {
        assert!((prob_to_odds(0.5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_prob_to_odds_clamped() {
        // Extreme values get clamped
        let odds = prob_to_odds(0.0);
        assert!(odds > 0.0);
        let odds = prob_to_odds(1.0);
        assert!(odds.is_finite());
    }

    #[test]
    fn test_odds_to_prob_zero() {
        assert!((odds_to_prob(0.0) - 0.001).abs() < 1e-9);
    }

    // --- dampen_lr ---

    #[test]
    fn test_dampen_full_confidence() {
        let lr = 2.0;
        let dampened = dampen_lr(lr, 1.0);
        assert!((dampened - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_dampen_zero_confidence() {
        let lr = 2.0;
        let dampened = dampen_lr(lr, 0.0);
        assert!(
            (dampened - 1.0).abs() < 1e-9,
            "zero confidence should give LR=1.0"
        );
    }

    #[test]
    fn test_dampen_half_confidence() {
        let lr = 4.0;
        let dampened = dampen_lr(lr, 0.5);
        // 4^0.5 = 2.0
        assert!((dampened - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_dampen_lr_less_than_one() {
        // LR < 1.0 (evidence against YES) should also dampen toward 1.0
        let lr = 0.25;
        let dampened = dampen_lr(lr, 0.5);
        // 0.25^0.5 = 0.5
        assert!((dampened - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_dampen_negative_lr() {
        assert!((dampen_lr(-1.0, 0.8) - 1.0).abs() < 1e-9);
    }

    // --- bayesian_update ---

    #[test]
    fn test_update_no_assessments() {
        let est = bayesian_update(0.5, &[]);
        assert!((est.posterior - 0.5).abs() < 1e-9);
        assert!((est.combined_lr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_update_single_agent_favors_yes() {
        let assessments = vec![AgentAssessment {
            role: "catalyst".into(),
            likelihood_ratio: 2.0,
            confidence: 1.0,
            reasoning: "Strong news".into(),
        }];
        let est = bayesian_update(0.5, &assessments);
        // prior_odds = 1.0, posterior_odds = 2.0, posterior = 2/3
        assert!((est.posterior - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_single_agent_favors_no() {
        let assessments = vec![AgentAssessment {
            role: "skeptic".into(),
            likelihood_ratio: 0.5,
            confidence: 1.0,
            reasoning: "Already priced in".into(),
        }];
        let est = bayesian_update(0.5, &assessments);
        // prior_odds = 1.0, posterior_odds = 0.5, posterior = 1/3
        assert!((est.posterior - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_uninformative_lr() {
        let assessments = vec![AgentAssessment {
            role: "skeptic".into(),
            likelihood_ratio: 1.0,
            confidence: 0.9,
            reasoning: "No change".into(),
        }];
        let est = bayesian_update(0.6, &assessments);
        assert!(
            (est.posterior - 0.6).abs() < 1e-6,
            "LR=1 should not move posterior"
        );
    }

    #[test]
    fn test_update_two_agents_agree() {
        let assessments = vec![
            AgentAssessment {
                role: "catalyst".into(),
                likelihood_ratio: 2.0,
                confidence: 1.0,
                reasoning: "a".into(),
            },
            AgentAssessment {
                role: "base_rate".into(),
                likelihood_ratio: 2.0,
                confidence: 1.0,
                reasoning: "b".into(),
            },
        ];
        let est = bayesian_update(0.5, &assessments);
        // prior_odds = 1.0, combined LR = 4.0, posterior_odds = 4.0, posterior = 4/5
        assert!((est.posterior - 0.8).abs() < 1e-6);
        assert!((est.combined_lr - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_agents_disagree() {
        let assessments = vec![
            AgentAssessment {
                role: "catalyst".into(),
                likelihood_ratio: 3.0,
                confidence: 0.8,
                reasoning: "bullish".into(),
            },
            AgentAssessment {
                role: "skeptic".into(),
                likelihood_ratio: 0.5,
                confidence: 0.8,
                reasoning: "bearish".into(),
            },
        ];
        let est = bayesian_update(0.5, &assessments);
        // LRs partially cancel. Agreement should be low -> confidence crushed.
        assert!(
            est.confidence < 0.5,
            "disagreement should reduce confidence: {}",
            est.confidence
        );
    }

    #[test]
    fn test_update_confidence_dampens_lr() {
        let full_conf = vec![AgentAssessment {
            role: "catalyst".into(),
            likelihood_ratio: 4.0,
            confidence: 1.0,
            reasoning: "sure".into(),
        }];
        let half_conf = vec![AgentAssessment {
            role: "catalyst".into(),
            likelihood_ratio: 4.0,
            confidence: 0.5,
            reasoning: "uncertain".into(),
        }];
        let est_full = bayesian_update(0.5, &full_conf);
        let est_half = bayesian_update(0.5, &half_conf);
        // Half confidence should move posterior less
        assert!(
            (est_full.posterior - 0.5).abs() > (est_half.posterior - 0.5).abs(),
            "half confidence should move less: full={:.3} half={:.3}",
            est_full.posterior,
            est_half.posterior
        );
    }

    #[test]
    fn test_update_asymmetric_prior() {
        let assessments = vec![AgentAssessment {
            role: "catalyst".into(),
            likelihood_ratio: 2.0,
            confidence: 1.0,
            reasoning: "news".into(),
        }];
        // Prior at 0.8 (strong YES already)
        let est = bayesian_update(0.8, &assessments);
        // prior_odds = 4.0, posterior_odds = 8.0, posterior = 8/9 ≈ 0.889
        assert!((est.posterior - 8.0 / 9.0).abs() < 1e-6);
        // Same LR moves a high prior less than a 50/50 prior
        let est_50 = bayesian_update(0.5, &assessments);
        let move_80 = est.posterior - 0.8;
        let move_50 = est_50.posterior - 0.5;
        assert!(move_80 < move_50, "high prior should move less");
    }

    #[test]
    fn test_update_three_agents_stacking() {
        let assessments = vec![
            AgentAssessment {
                role: "a".into(),
                likelihood_ratio: 1.5,
                confidence: 1.0,
                reasoning: "".into(),
            },
            AgentAssessment {
                role: "b".into(),
                likelihood_ratio: 1.5,
                confidence: 1.0,
                reasoning: "".into(),
            },
            AgentAssessment {
                role: "c".into(),
                likelihood_ratio: 1.5,
                confidence: 1.0,
                reasoning: "".into(),
            },
        ];
        let est = bayesian_update(0.5, &assessments);
        // combined LR = 1.5^3 = 3.375
        assert!((est.combined_lr - 3.375).abs() < 1e-6);
        // posterior_odds = 3.375, posterior = 3.375/4.375
        let expected = 3.375 / 4.375;
        assert!((est.posterior - expected).abs() < 1e-6);
    }

    #[test]
    fn test_posterior_clamped() {
        // Extreme LR shouldn't push posterior to 0 or 1
        let assessments = vec![AgentAssessment {
            role: "test".into(),
            likelihood_ratio: 10000.0,
            confidence: 1.0,
            reasoning: "".into(),
        }];
        let est = bayesian_update(0.99, &assessments);
        assert!(est.posterior < 1.0);
        assert!(est.posterior > 0.0);
    }

    // --- compute_agreement ---

    #[test]
    fn test_agreement_same_direction() {
        let assessments = vec![
            AgentAssessment {
                role: "a".into(),
                likelihood_ratio: 2.0,
                confidence: 1.0,
                reasoning: "".into(),
            },
            AgentAssessment {
                role: "b".into(),
                likelihood_ratio: 2.5,
                confidence: 1.0,
                reasoning: "".into(),
            },
        ];
        let agreement = compute_agreement(&assessments);
        assert!(
            agreement > 0.8,
            "same direction should have high agreement: {agreement}"
        );
    }

    #[test]
    fn test_agreement_opposite_directions() {
        let assessments = vec![
            AgentAssessment {
                role: "a".into(),
                likelihood_ratio: 3.0,
                confidence: 1.0,
                reasoning: "".into(),
            },
            AgentAssessment {
                role: "b".into(),
                likelihood_ratio: 0.3,
                confidence: 1.0,
                reasoning: "".into(),
            },
        ];
        let agreement = compute_agreement(&assessments);
        assert!(
            agreement < 0.5,
            "opposite directions should have low agreement: {agreement}"
        );
    }

    #[test]
    fn test_agreement_single_agent() {
        let assessments = vec![AgentAssessment {
            role: "a".into(),
            likelihood_ratio: 5.0,
            confidence: 1.0,
            reasoning: "".into(),
        }];
        assert!((compute_agreement(&assessments) - 1.0).abs() < 1e-9);
    }

    // --- compute_edge ---

    #[test]
    fn test_edge_yes_side() {
        let est = BayesianEstimate {
            prior: 0.5,
            posterior: 0.7,
            combined_lr: 2.33,
            confidence: 0.8,
            reasoning: String::new(),
        };
        let (is_yes, edge, price, prob) = compute_edge(&est, 0.5).unwrap();
        assert!(is_yes);
        assert!((edge - 0.2).abs() < 1e-9);
        assert!((price - 0.5).abs() < 1e-9);
        assert!((prob - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_edge_no_side() {
        let est = BayesianEstimate {
            prior: 0.5,
            posterior: 0.3,
            combined_lr: 0.43,
            confidence: 0.8,
            reasoning: String::new(),
        };
        let (is_yes, edge, price, prob) = compute_edge(&est, 0.5).unwrap();
        assert!(!is_yes);
        assert!((edge - 0.2).abs() < 1e-9);
        assert!((price - 0.5).abs() < 1e-9);
        assert!((prob - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_edge_none_when_no_edge() {
        let est = BayesianEstimate {
            prior: 0.5,
            posterior: 0.5,
            combined_lr: 1.0,
            confidence: 0.5,
            reasoning: String::new(),
        };
        assert!(compute_edge(&est, 0.5).is_none());
    }

    #[test]
    fn test_edge_small_yes_edge() {
        let est = BayesianEstimate {
            prior: 0.5,
            posterior: 0.52,
            combined_lr: 1.08,
            confidence: 0.6,
            reasoning: String::new(),
        };
        let (is_yes, edge, _, _) = compute_edge(&est, 0.5).unwrap();
        assert!(is_yes);
        assert!((edge - 0.02).abs() < 1e-9);
    }

    // --- reasoning formatting ---

    #[test]
    fn test_reasoning_includes_agents() {
        let assessments = vec![
            AgentAssessment {
                role: "skeptic".into(),
                likelihood_ratio: 0.8,
                confidence: 0.7,
                reasoning: "priced in".into(),
            },
            AgentAssessment {
                role: "catalyst".into(),
                likelihood_ratio: 1.5,
                confidence: 0.9,
                reasoning: "breaking".into(),
            },
        ];
        let est = bayesian_update(0.5, &assessments);
        assert!(est.reasoning.contains("[skeptic"));
        assert!(est.reasoning.contains("[catalyst"));
        assert!(est.reasoning.contains("Bayes:"));
    }
}
