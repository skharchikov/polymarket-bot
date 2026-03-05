/// Kelly Criterion position sizing.
///
/// Returns the fraction of bankroll to wager.
/// `prob` = estimated true probability of YES outcome
/// `market_price` = current YES price (cost to buy)
pub fn kelly_fraction(prob: f64, market_price: f64) -> f64 {
    if market_price <= 0.0 || market_price >= 1.0 || prob <= 0.0 || prob >= 1.0 {
        return 0.0;
    }

    // Odds: if you buy YES at `market_price`, you win (1 - market_price) per unit risked
    let b = (1.0 - market_price) / market_price;
    let q = 1.0 - prob;

    let f = (b * prob - q) / b;
    f.max(0.0) // never suggest negative sizing
}

/// Fractional Kelly for safer sizing.
pub fn fractional_kelly(prob: f64, market_price: f64, fraction: f64) -> f64 {
    kelly_fraction(prob, market_price) * fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_edge_returns_zero() {
        let f = kelly_fraction(0.60, 0.60);
        assert!(f < 0.01);
    }

    #[test]
    fn test_positive_edge() {
        let f = kelly_fraction(0.70, 0.60);
        assert!(f > 0.0);
    }

    #[test]
    fn test_fractional_kelly_is_smaller() {
        let full = kelly_fraction(0.70, 0.60);
        let quarter = fractional_kelly(0.70, 0.60, 0.25);
        assert!(quarter < full);
        assert!((quarter - full * 0.25).abs() < 1e-10);
    }
}
