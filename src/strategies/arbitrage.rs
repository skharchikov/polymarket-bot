use super::signal::{Side, Signal, SignalSource};

/// Detect mispricing between logically linked markets.
/// If two markets represent complementary outcomes, their YES prices should sum to ~1.0.
pub fn find_pair_arbitrage(
    market_a_id: &str,
    market_b_id: &str,
    price_a: f64,
    price_b: f64,
    threshold: f64,
) -> Vec<Signal> {
    let total = price_a + price_b;
    let spread = total - 1.0;

    if spread.abs() < threshold {
        return vec![];
    }

    let mut signals = Vec::new();

    if spread > threshold {
        // Overpriced total: sell the more expensive, buy the cheaper
        // In prediction markets, "sell YES" = "buy NO"
        let (expensive_id, cheap_id, expensive_price, _cheap_price) = if price_a > price_b {
            (market_a_id, market_b_id, price_a, price_b)
        } else {
            (market_b_id, market_a_id, price_b, price_a)
        };

        signals.push(Signal {
            market_id: expensive_id.to_string(),
            side: Side::No,
            edge: spread / 2.0,
            confidence: 1.0, // arbitrage is high confidence
            size_pct: 0.03,
            source: SignalSource::Arbitrage {
                paired_market: cheap_id.to_string(),
                spread,
            },
        });
        signals.push(Signal {
            market_id: cheap_id.to_string(),
            side: Side::Yes,
            edge: spread / 2.0,
            confidence: 1.0,
            size_pct: 0.03,
            source: SignalSource::Arbitrage {
                paired_market: expensive_id.to_string(),
                spread,
            },
        });
    } else if spread < -threshold {
        // Underpriced total: buy both
        signals.push(Signal {
            market_id: market_a_id.to_string(),
            side: Side::Yes,
            edge: spread.abs() / 2.0,
            confidence: 1.0,
            size_pct: 0.03,
            source: SignalSource::Arbitrage {
                paired_market: market_b_id.to_string(),
                spread,
            },
        });
        signals.push(Signal {
            market_id: market_b_id.to_string(),
            side: Side::Yes,
            edge: spread.abs() / 2.0,
            confidence: 1.0,
            size_pct: 0.03,
            source: SignalSource::Arbitrage {
                paired_market: market_a_id.to_string(),
                spread,
            },
        });
    }

    signals
}

/// For a single market, check if YES + NO prices deviate from 1.0.
/// This catches internal spread arbitrage.
pub fn find_spread_arbitrage(
    market_id: &str,
    yes_price: f64,
    no_price: f64,
    threshold: f64,
) -> Option<Signal> {
    let total = yes_price + no_price;
    let deviation = total - 1.0;

    if deviation.abs() < threshold {
        return None;
    }

    // If total > 1: both sides overpriced, sell the more expensive
    // If total < 1: both sides underpriced, buy the cheaper
    let (side, edge) = if deviation > 0.0 {
        if yes_price > no_price {
            (Side::No, deviation)
        } else {
            (Side::Yes, deviation)
        }
    } else {
        if yes_price < no_price {
            (Side::Yes, deviation.abs())
        } else {
            (Side::No, deviation.abs())
        }
    };

    Some(Signal {
        market_id: market_id.to_string(),
        side,
        edge,
        confidence: 0.95,
        size_pct: 0.02,
        source: SignalSource::Arbitrage {
            paired_market: market_id.to_string(),
            spread: deviation,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_arbitrage_when_balanced() {
        let signals = find_pair_arbitrage("a", "b", 0.60, 0.40, 0.03);
        assert!(signals.is_empty());
    }

    #[test]
    fn test_overpriced_pair() {
        let signals = find_pair_arbitrage("a", "b", 0.70, 0.40, 0.03);
        assert_eq!(signals.len(), 2);
    }

    #[test]
    fn test_spread_arb() {
        let sig = find_spread_arbitrage("m1", 0.55, 0.55, 0.03);
        assert!(sig.is_some());
        assert_eq!(sig.unwrap().side, Side::Yes); // total > 1, both equal, buy the cheaper (yes)
    }
}
