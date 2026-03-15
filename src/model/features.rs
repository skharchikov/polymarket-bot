#![allow(dead_code)]

use crate::data::models::{GammaMarket, PriceTick};

/// Order book statistics computed from CLOB book endpoint.
#[derive(Debug, Clone, Default)]
pub struct OrderBookStats {
    pub depth: f64,
    pub order_imbalance: f64,
    pub spread: f64,
}

/// Feature vector matching the Python training pipeline's FEATURE_COLS.
/// Order must be identical to scripts/train_model.py.
///
/// Removed from v1: news_count, best_news_score, avg_news_age_hours,
/// order_imbalance, spread — these were always 0 in training (not available
/// retroactively for historical markets) but non-zero in live inference,
/// causing distribution shift with zero learning signal.
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    pub yes_price: f64,
    pub momentum_1h: f64,
    pub momentum_24h: f64,
    pub volatility_24h: f64,
    pub rsi: f64,
    pub log_volume: f64,
    pub log_liquidity: f64,
    pub days_to_expiry: f64,
    pub is_crypto: f64,
    pub is_politics: f64,
    pub is_sports: f64,
    // Gamma API price changes (more reliable than computed momentum)
    pub price_change_1d: f64,
    pub price_change_1w: f64,
}

impl MarketFeatures {
    /// Feature names in the same order as the Python pipeline.
    pub const NAMES: &[&str] = &[
        "yes_price",
        "momentum_1h",
        "momentum_24h",
        "volatility_24h",
        "rsi",
        "log_volume",
        "log_liquidity",
        "days_to_expiry",
        "is_crypto",
        "is_politics",
        "is_sports",
        "price_change_1d",
        "price_change_1w",
    ];

    /// Convert to fixed-order f64 vector for model input.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.yes_price,
            self.momentum_1h,
            self.momentum_24h,
            self.volatility_24h,
            self.rsi,
            self.log_volume,
            self.log_liquidity,
            self.days_to_expiry,
            self.is_crypto,
            self.is_politics,
            self.is_sports,
            self.price_change_1d,
            self.price_change_1w,
        ]
    }

    /// Build features from a market, price history, and matched news.
    /// News signals are passed through for logging/display but are no longer
    /// part of the model feature vector (see struct comment).
    pub fn from_market_and_news(
        market: &GammaMarket,
        current_price: f64,
        history: &[PriceTick],
        _news_count: usize,
        _best_news_score: f64,
        _avg_news_age_hours: f64,
    ) -> Self {
        Self::from_market_and_history(market, current_price, history)
    }

    /// Build features from a market and its price history (no news data).
    pub fn from_market_and_history(
        market: &GammaMarket,
        current_price: f64,
        history: &[PriceTick],
    ) -> Self {
        let now_ts = chrono::Utc::now().timestamp();

        // Price lookbacks
        let p_1h = price_at_offset(history, now_ts, 3600);
        let p_24h = price_at_offset(history, now_ts, 86400);

        let momentum_1h = p_1h.map(|p| current_price - p).unwrap_or(0.0);
        let momentum_24h = p_24h.map(|p| current_price - p).unwrap_or(0.0);

        // Volatility: std of returns over recent ticks
        let volatility_24h = compute_volatility(history, 24);

        // RSI (14-period)
        let rsi = compute_rsi(history, 14);

        // Volume/liquidity
        let log_volume = (market.volume_num + 1.0).ln();
        let log_liquidity = (market.liquidity_num + 1.0).ln();

        // Days to expiry
        let days_to_expiry = market
            .end_date
            .as_ref()
            .and_then(|d| chrono::DateTime::parse_from_rfc3339(d).ok())
            .map(|end| {
                let remaining = end.timestamp() - now_ts;
                (remaining as f64 / 86400.0).max(0.0)
            })
            .unwrap_or(30.0);

        // Category flags
        let cat = market
            .category
            .as_deref()
            .unwrap_or_default()
            .to_lowercase();
        let q = market.question.to_lowercase();

        let is_crypto = if market.is_crypto_related() { 1.0 } else { 0.0 };
        let is_politics = if cat.contains("politic")
            || q.contains("election")
            || q.contains("president")
            || q.contains("vote")
        {
            1.0
        } else {
            0.0
        };
        let is_sports = if cat.contains("sport")
            || q.contains("nba")
            || q.contains("nfl")
            || q.contains("soccer")
        {
            1.0
        } else {
            0.0
        };

        // Gamma API price changes (fallback to computed momentum)
        let price_change_1d = market.one_day_price_change.unwrap_or(momentum_24h);
        let price_change_1w = market.one_week_price_change.unwrap_or(0.0);

        Self {
            yes_price: current_price,
            momentum_1h,
            momentum_24h,
            volatility_24h,
            rsi,
            log_volume,
            log_liquidity,
            days_to_expiry,
            is_crypto,
            is_politics,
            is_sports,
            price_change_1d,
            price_change_1w,
        }
    }
}

/// Find price approximately `offset_secs` before the last tick.
fn price_at_offset(history: &[PriceTick], now_ts: i64, offset_secs: i64) -> Option<f64> {
    if history.is_empty() {
        return None;
    }
    let target = now_ts - offset_secs;
    let idx = history.partition_point(|t| t.t < target);
    if idx < history.len() {
        Some(history[idx].p)
    } else {
        history.last().map(|t| t.p)
    }
}

/// Compute volatility (std of returns) over the last `hours` of ticks.
fn compute_volatility(history: &[PriceTick], hours: usize) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let lookback = history.len().min(hours);
    let window = &history[history.len() - lookback..];
    if window.len() < 2 {
        return 0.0;
    }

    let returns: Vec<f64> = window
        .windows(2)
        .map(|w| (w[1].p - w[0].p) / (w[0].p + 1e-10))
        .collect();

    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    variance.sqrt()
}

/// Compute RSI on 0-1 scale.
fn compute_rsi(history: &[PriceTick], period: usize) -> f64 {
    if history.len() < period + 1 {
        return 0.5;
    }
    let recent = &history[history.len() - period - 1..];
    let mut gains = 0.0;
    let mut losses = 0.0;

    for w in recent.windows(2) {
        let delta = w[1].p - w[0].p;
        if delta > 0.0 {
            gains += delta;
        } else {
            losses -= delta;
        }
    }

    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;

    if avg_loss == 0.0 {
        return 1.0;
    }
    let rs = avg_gain / avg_loss;
    1.0 - (1.0 / (1.0 + rs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ticks(prices: &[f64], start_ts: i64, interval: i64) -> Vec<PriceTick> {
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| PriceTick {
                t: start_ts + i as i64 * interval,
                p,
            })
            .collect()
    }

    #[test]
    fn test_rsi_all_gains() {
        let ticks = make_ticks(
            &[
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            ],
            0,
            3600,
        );
        let rsi = compute_rsi(&ticks, 14);
        assert!(rsi > 0.5, "RSI with mostly gains should be > 0.5: {rsi}");
    }

    #[test]
    fn test_rsi_all_losses() {
        let ticks = make_ticks(
            &[
                0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4,
            ],
            0,
            3600,
        );
        let rsi = compute_rsi(&ticks, 14);
        assert!(rsi < 0.5, "RSI with mostly losses should be < 0.5: {rsi}");
    }

    #[test]
    fn test_volatility_flat() {
        let ticks = make_ticks(&[0.5; 20], 0, 3600);
        let vol = compute_volatility(&ticks, 24);
        assert!(
            vol < 0.001,
            "Flat prices should have near-zero volatility: {vol}"
        );
    }

    #[test]
    fn test_volatility_high() {
        let prices: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 0.3 } else { 0.7 })
            .collect();
        let ticks = make_ticks(&prices, 0, 3600);
        let vol = compute_volatility(&ticks, 24);
        assert!(
            vol > 0.1,
            "Oscillating prices should have high volatility: {vol}"
        );
    }

    #[test]
    fn test_features_to_vec_length() {
        let features = MarketFeatures {
            yes_price: 0.5,
            momentum_1h: 0.01,
            momentum_24h: -0.02,
            volatility_24h: 0.05,
            rsi: 0.55,
            log_volume: 10.0,
            log_liquidity: 8.0,
            days_to_expiry: 15.0,
            is_crypto: 1.0,
            is_politics: 0.0,
            is_sports: 0.0,
            price_change_1d: 0.03,
            price_change_1w: -0.05,
        };
        assert_eq!(features.to_vec().len(), MarketFeatures::NAMES.len());
    }

    #[test]
    fn test_price_at_offset() {
        let ticks = make_ticks(&[0.3, 0.4, 0.5, 0.6, 0.7], 1000, 3600);
        let p = price_at_offset(&ticks, 5000, 7200);
        assert!(p.is_some());
    }
}
