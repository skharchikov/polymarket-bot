use anyhow::Result;
use sqlx::PgPool;

const NUM_BINS: usize = 10;
const BIN_WIDTH: f64 = 1.0 / NUM_BINS as f64;

/// Calibration curve built from historical LLM estimates vs actual outcomes.
pub struct CalibrationCurve {
    /// For each bin [0..NUM_BINS], the calibrated actual outcome rate.
    /// Uses Laplace smoothing: (wins + 1) / (total + 2).
    bins: [f64; NUM_BINS],
    /// Whether enough data exists to apply correction.
    pub active: bool,
    pub total_samples: usize,
}

impl CalibrationCurve {
    /// Load calibration data from resolved LLM estimates.
    pub async fn load(pool: &PgPool, min_samples: usize) -> Result<Self> {
        let rows: Vec<(f64, bool)> = sqlx::query_as(
            "SELECT raw_probability, outcome \
             FROM llm_estimates \
             WHERE resolved = true AND agent_role = 'consensus' AND outcome IS NOT NULL",
        )
        .fetch_all(pool)
        .await?;

        let total_samples = rows.len();
        let active = total_samples >= min_samples;

        let mut bin_wins = [0u32; NUM_BINS];
        let mut bin_total = [0u32; NUM_BINS];

        for (prob, won) in &rows {
            let idx = prob_to_bin(*prob);
            bin_total[idx] += 1;
            if *won {
                bin_wins[idx] += 1;
            }
        }

        let mut bins = [0.0; NUM_BINS];
        for i in 0..NUM_BINS {
            // Laplace smoothing: prior is the bin midpoint
            let midpoint = (i as f64 + 0.5) * BIN_WIDTH;
            if bin_total[i] == 0 {
                bins[i] = midpoint;
            } else {
                bins[i] = (bin_wins[i] as f64 + 1.0) / (bin_total[i] as f64 + 2.0);
            }
        }

        if active {
            tracing::info!(
                samples = total_samples,
                "Calibration curve loaded and active"
            );
        } else {
            tracing::info!(
                samples = total_samples,
                min = min_samples,
                "Calibration curve inactive (insufficient data)"
            );
        }

        Ok(Self {
            bins,
            active,
            total_samples,
        })
    }

    /// Apply calibration correction to a raw probability estimate.
    /// Returns the raw value unchanged if calibration is inactive.
    pub fn correct(&self, raw_prob: f64) -> f64 {
        if !self.active {
            return raw_prob;
        }

        let idx = prob_to_bin(raw_prob);
        let bin_center = (idx as f64 + 0.5) * BIN_WIDTH;
        let calibrated = self.bins[idx];

        // Interpolate with neighboring bin for smoother output
        let offset = raw_prob - bin_center;
        if offset > 0.0 && idx + 1 < NUM_BINS {
            let next = self.bins[idx + 1];
            let weight = offset / BIN_WIDTH;
            lerp(calibrated, next, weight)
        } else if offset < 0.0 && idx > 0 {
            let prev = self.bins[idx - 1];
            let weight = -offset / BIN_WIDTH;
            lerp(calibrated, prev, weight)
        } else {
            calibrated
        }
        .clamp(0.01, 0.99)
    }

    /// Summary string for LLM prompt context.
    pub fn summary(&self) -> String {
        if !self.active {
            return format!(
                "Calibration: inactive ({} samples, need more data)",
                self.total_samples
            );
        }

        let mut lines = vec![format!(
            "CALIBRATION DATA ({} resolved estimates):",
            self.total_samples
        )];
        for i in 0..NUM_BINS {
            let lo = i as f64 * BIN_WIDTH * 100.0;
            let hi = (i + 1) as f64 * BIN_WIDTH * 100.0;
            let actual = self.bins[i] * 100.0;
            let expected = (i as f64 + 0.5) * BIN_WIDTH * 100.0;
            let diff = actual - expected;
            let label = if diff.abs() < 3.0 {
                "well-calibrated"
            } else if diff > 0.0 {
                "UNDERCONFIDENT"
            } else {
                "OVERCONFIDENT"
            };
            lines.push(format!(
                "  {lo:.0}-{hi:.0}%: actual={actual:.0}% ({diff:+.0}%) {label}"
            ));
        }
        lines.join("\n")
    }
}

fn prob_to_bin(p: f64) -> usize {
    let idx = (p / BIN_WIDTH) as usize;
    idx.min(NUM_BINS - 1)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prob_to_bin() {
        assert_eq!(prob_to_bin(0.0), 0);
        assert_eq!(prob_to_bin(0.05), 0);
        assert_eq!(prob_to_bin(0.15), 1);
        assert_eq!(prob_to_bin(0.95), 9);
        assert_eq!(prob_to_bin(1.0), 9);
    }

    #[test]
    fn test_inactive_returns_raw() {
        let curve = CalibrationCurve {
            bins: [0.0; NUM_BINS],
            active: false,
            total_samples: 0,
        };
        assert_eq!(curve.correct(0.7), 0.7);
        assert_eq!(curve.correct(0.1), 0.1);
        assert_eq!(curve.correct(0.99), 0.99);
    }

    /// Build a perfectly calibrated curve (identity) where bins[i] = midpoint.
    fn identity_curve() -> CalibrationCurve {
        let mut bins = [0.0; NUM_BINS];
        for (i, bin) in bins.iter_mut().enumerate() {
            *bin = (i as f64 + 0.5) * BIN_WIDTH;
        }
        CalibrationCurve {
            bins,
            active: true,
            total_samples: 100,
        }
    }

    #[test]
    fn test_identity_curve_returns_close_to_input() {
        let curve = identity_curve();
        // At bin centers, correction should be exact
        for i in 0..NUM_BINS {
            let center = (i as f64 + 0.5) * BIN_WIDTH;
            let corrected = curve.correct(center);
            assert!(
                (corrected - center).abs() < 0.01,
                "At center {center}: got {corrected}"
            );
        }
    }

    #[test]
    fn test_overconfident_curve_shifts_down() {
        // LLM says 70% but actual is 55% → overconfident in high bins
        let mut bins = [0.0; NUM_BINS];
        for (i, bin) in bins.iter_mut().enumerate() {
            let midpoint = (i as f64 + 0.5) * BIN_WIDTH;
            // Shift actuals toward 50% (overconfident model)
            *bin = midpoint * 0.7 + 0.15;
        }
        let curve = CalibrationCurve {
            bins,
            active: true,
            total_samples: 100,
        };

        // High raw prob should be corrected downward
        let corrected = curve.correct(0.85);
        assert!(
            corrected < 0.85,
            "0.85 should be corrected down, got {corrected}"
        );

        // Low raw prob should be corrected upward
        let corrected_low = curve.correct(0.15);
        assert!(
            corrected_low > 0.15,
            "0.15 should be corrected up, got {corrected_low}"
        );
    }

    #[test]
    fn test_correct_clamps_output() {
        // Extreme bins that would produce out-of-range values
        let mut bins = [0.0; NUM_BINS];
        bins[0] = 0.001; // Very low actual rate in 0-10% bin
        bins[9] = 0.999; // Very high actual rate in 90-100% bin
        for (i, bin) in bins.iter_mut().enumerate().take(9).skip(1) {
            *bin = (i as f64 + 0.5) * BIN_WIDTH;
        }
        let curve = CalibrationCurve {
            bins,
            active: true,
            total_samples: 100,
        };

        let low = curve.correct(0.01);
        assert!(low >= 0.01, "Should be clamped to >=0.01, got {low}");
        let high = curve.correct(0.99);
        assert!(high <= 0.99, "Should be clamped to <=0.99, got {high}");
    }

    #[test]
    fn test_interpolation_smoothness() {
        let curve = identity_curve();
        // Adjacent values should produce similar corrections
        let a = curve.correct(0.50);
        let b = curve.correct(0.51);
        assert!(
            (a - b).abs() < 0.05,
            "Adjacent inputs should give similar outputs: {a} vs {b}"
        );
    }

    #[test]
    fn test_summary_inactive() {
        let curve = CalibrationCurve {
            bins: [0.0; NUM_BINS],
            active: false,
            total_samples: 5,
        };
        let s = curve.summary();
        assert!(s.contains("inactive"));
        assert!(s.contains("5 samples"));
    }

    #[test]
    fn test_summary_active_shows_bins() {
        let curve = identity_curve();
        let s = curve.summary();
        assert!(s.contains("CALIBRATION DATA"));
        assert!(s.contains("100 resolved estimates"));
        // Should have 10 bin lines
        assert!(s.contains("0-10%"));
        assert!(s.contains("90-100%"));
    }

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 1.0, 0.0) - 0.0).abs() < 1e-9);
        assert!((lerp(0.0, 1.0, 1.0) - 1.0).abs() < 1e-9);
        assert!((lerp(0.0, 1.0, 0.5) - 0.5).abs() < 1e-9);
        assert!((lerp(2.0, 4.0, 0.25) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_prob_to_bin_boundary() {
        // Exact bin boundaries
        assert_eq!(prob_to_bin(0.1), 1);
        assert_eq!(prob_to_bin(0.5), 5);
        assert_eq!(prob_to_bin(0.099), 0);
    }

    #[test]
    fn test_correct_monotonic() {
        // With identity curve, correction should be roughly monotonic
        let curve = identity_curve();
        let mut prev = 0.0;
        for i in 1..=19 {
            let p = i as f64 * 0.05;
            let c = curve.correct(p);
            assert!(c >= prev - 0.01, "Non-monotonic at {p}: {c} < {prev}");
            prev = c;
        }
    }

    #[test]
    fn test_underconfident_curve_shifts_up() {
        // Actual outcomes are higher than LLM estimates
        let mut bins = [0.0; NUM_BINS];
        for (i, bin) in bins.iter_mut().enumerate() {
            let midpoint = (i as f64 + 0.5) * BIN_WIDTH;
            *bin = midpoint * 0.5 + 0.35; // shifts actuals upward
        }
        let curve = CalibrationCurve {
            bins,
            active: true,
            total_samples: 100,
        };
        let corrected = curve.correct(0.35);
        assert!(
            corrected > 0.35,
            "0.35 should be corrected up, got {corrected}"
        );
    }
}
