#![allow(dead_code)]

use crate::data::models::{GammaMarket, PriceTick};
use serde::Serialize;

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
///
/// v2 additions: days_since_created, created_to_expiry_span (temporal features).
/// Category flag (is_crypto) is sent as binary 0/1 from Rust;
/// the Python sidecar applies target encoding before model inference.
///
/// v3: removed log_liquidity, is_politics, is_sports — zero SHAP importance.
/// v4: re-added is_sports (ADR 009) — training data now contains sports markets,
///     previously zero SHAP because training data had none. 77% of live bets are
///     sports; model needs this signal to learn to discount them.
/// v5: added 16 NLP features from question text (no price leakage).
///     Inspired by NavnoorBawa (88-92% accuracy on high-confidence predictions).
#[derive(Debug, Clone, Serialize)]
pub struct MarketFeatures {
    pub yes_price: f64,
    pub momentum_1h: f64,
    pub momentum_24h: f64,
    pub volatility_24h: f64,
    pub rsi: f64,
    pub log_volume: f64,
    pub days_to_expiry: f64,
    pub is_crypto: f64,
    // Gamma API price changes (more reliable than computed momentum)
    pub price_change_1d: f64,
    pub price_change_1w: f64,
    // Temporal features (v2)
    pub days_since_created: f64,
    pub created_to_expiry_span: f64,
    // Category flag (v4, re-added) — sent as binary 0/1, target-encoded by sidecar
    pub is_sports: f64,
    // NLP features (v5) — extracted from question text
    pub q_length: f64,
    pub q_word_count: f64,
    pub q_avg_word_len: f64,
    pub q_word_diversity: f64,
    pub q_has_number: f64,
    pub q_has_year: f64,
    pub q_has_percent: f64,
    pub q_has_dollar: f64,
    pub q_has_date: f64,
    pub q_starts_will: f64,
    pub q_has_by: f64,
    pub q_has_before: f64,
    pub q_has_above: f64,
    pub q_sentiment_pos: f64,
    pub q_sentiment_neg: f64,
    pub q_certainty: f64,
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
        "days_to_expiry",
        "is_crypto",
        "price_change_1d",
        "price_change_1w",
        "days_since_created",
        "created_to_expiry_span",
        "is_sports",
        "q_length",
        "q_word_count",
        "q_avg_word_len",
        "q_word_diversity",
        "q_has_number",
        "q_has_year",
        "q_has_percent",
        "q_has_dollar",
        "q_has_date",
        "q_starts_will",
        "q_has_by",
        "q_has_before",
        "q_has_above",
        "q_sentiment_pos",
        "q_sentiment_neg",
        "q_certainty",
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
            self.days_to_expiry,
            self.is_crypto,
            self.price_change_1d,
            self.price_change_1w,
            self.days_since_created,
            self.created_to_expiry_span,
            self.is_sports,
            self.q_length,
            self.q_word_count,
            self.q_avg_word_len,
            self.q_word_diversity,
            self.q_has_number,
            self.q_has_year,
            self.q_has_percent,
            self.q_has_dollar,
            self.q_has_date,
            self.q_starts_will,
            self.q_has_by,
            self.q_has_before,
            self.q_has_above,
            self.q_sentiment_pos,
            self.q_sentiment_neg,
            self.q_certainty,
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

        // Volume
        let log_volume = (market.volume_num + 1.0).ln();

        // Days to expiry
        let end_ts = market
            .end_date
            .as_ref()
            .and_then(|d| chrono::DateTime::parse_from_rfc3339(d).ok())
            .map(|d| d.timestamp());

        let days_to_expiry = end_ts
            .map(|end| ((end - now_ts) as f64 / 86400.0).max(0.0))
            .unwrap_or(30.0);

        // Temporal features
        let created_ts = market
            .created_at
            .as_ref()
            .and_then(|d| chrono::DateTime::parse_from_rfc3339(d).ok())
            .map(|d| d.timestamp());

        let days_since_created = created_ts
            .map(|c| ((now_ts - c) as f64 / 86400.0).max(0.0))
            .unwrap_or(30.0);

        let created_to_expiry_span = match (created_ts, end_ts) {
            (Some(c), Some(e)) => ((e - c) as f64 / 86400.0).max(0.0),
            _ => 30.0,
        };

        // Category flags (sent as binary; sidecar applies target encoding)
        let is_crypto = if market.is_crypto_related() { 1.0 } else { 0.0 };
        let is_sports = if market.is_sports_or_esports() {
            1.0
        } else {
            0.0
        };

        // Gamma API price changes (fallback to computed momentum)
        let price_change_1d = market.one_day_price_change.unwrap_or(momentum_24h);
        let price_change_1w = market.one_week_price_change.unwrap_or(0.0);

        // NLP features from question text
        let nlp = extract_nlp_features(&market.question);

        Self {
            yes_price: current_price,
            momentum_1h,
            momentum_24h,
            volatility_24h,
            rsi,
            log_volume,
            days_to_expiry,
            is_crypto,
            price_change_1d,
            price_change_1w,
            days_since_created,
            created_to_expiry_span,
            is_sports,
            q_length: nlp.q_length,
            q_word_count: nlp.q_word_count,
            q_avg_word_len: nlp.q_avg_word_len,
            q_word_diversity: nlp.q_word_diversity,
            q_has_number: nlp.q_has_number,
            q_has_year: nlp.q_has_year,
            q_has_percent: nlp.q_has_percent,
            q_has_dollar: nlp.q_has_dollar,
            q_has_date: nlp.q_has_date,
            q_starts_will: nlp.q_starts_will,
            q_has_by: nlp.q_has_by,
            q_has_before: nlp.q_has_before,
            q_has_above: nlp.q_has_above,
            q_sentiment_pos: nlp.q_sentiment_pos,
            q_sentiment_neg: nlp.q_sentiment_neg,
            q_certainty: nlp.q_certainty,
        }
    }
}

struct NlpFeatures {
    q_length: f64,
    q_word_count: f64,
    q_avg_word_len: f64,
    q_word_diversity: f64,
    q_has_number: f64,
    q_has_year: f64,
    q_has_percent: f64,
    q_has_dollar: f64,
    q_has_date: f64,
    q_starts_will: f64,
    q_has_by: f64,
    q_has_before: f64,
    q_has_above: f64,
    q_sentiment_pos: f64,
    q_sentiment_neg: f64,
    q_certainty: f64,
}

fn extract_nlp_features(question: &str) -> NlpFeatures {
    let q = question.to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();
    let n_words = words.len().max(1) as f64;
    let unique: std::collections::HashSet<&str> = words.iter().copied().collect();

    let q_length = question.len() as f64;
    let q_word_count = n_words;
    let q_avg_word_len = words.iter().map(|w| w.len() as f64).sum::<f64>() / n_words;
    let q_word_diversity = unique.len() as f64 / n_words;

    let q_has_number = if q.chars().any(|c| c.is_ascii_digit()) {
        1.0
    } else {
        0.0
    };
    let q_has_year = if q.contains("202") { 1.0 } else { 0.0 };
    let q_has_percent = if q.contains('%') { 1.0 } else { 0.0 };
    let q_has_dollar = if q.contains('$') { 1.0 } else { 0.0 };

    let months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    let q_has_date = if months.iter().any(|m| q.contains(m)) || q.contains('/') {
        1.0
    } else {
        0.0
    };

    let q_starts_will = if q.starts_with("will ") { 1.0 } else { 0.0 };
    let q_has_by = if q.contains(" by ") { 1.0 } else { 0.0 };
    let q_has_before = if q.contains(" before ") { 1.0 } else { 0.0 };

    let above_words = ["above", "over", "exceed", "hit", "reach", "break"];
    let q_has_above = if above_words.iter().any(|w| q.contains(w)) {
        1.0
    } else {
        0.0
    };

    let pos_words = [
        "win", "pass", "above", "exceed", "achieve", "surge", "gain", "rise", "increase",
        "approve", "success", "agree", "accept", "hit", "reach",
    ];
    let neg_words = [
        "lose", "fail", "below", "crash", "reject", "decline", "fall", "drop", "decrease", "deny",
        "miss", "ban", "block", "cancel", "collapse",
    ];
    let cert_words = [
        "will",
        "definitely",
        "certainly",
        "must",
        "always",
        "guaranteed",
    ];

    let q_sentiment_pos = words.iter().filter(|w| pos_words.contains(w)).count() as f64;
    let q_sentiment_neg = words.iter().filter(|w| neg_words.contains(w)).count() as f64;
    let q_certainty = words.iter().filter(|w| cert_words.contains(w)).count() as f64;

    NlpFeatures {
        q_length,
        q_word_count,
        q_avg_word_len,
        q_word_diversity,
        q_has_number,
        q_has_year,
        q_has_percent,
        q_has_dollar,
        q_has_date,
        q_starts_will,
        q_has_by,
        q_has_before,
        q_has_above,
        q_sentiment_pos,
        q_sentiment_neg,
        q_certainty,
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

/// Compute RSI on 0-1 scale, time-weighted by the duration each price persisted.
fn compute_rsi(history: &[PriceTick], period: usize) -> f64 {
    if history.len() < period + 1 {
        return 0.5;
    }
    let recent = &history[history.len() - period - 1..];
    let mut gains = 0.0;
    let mut losses = 0.0;
    let mut total_time = 0.0;

    for w in recent.windows(2) {
        let delta = w[1].p - w[0].p;
        let dt = (w[1].t - w[0].t).max(1) as f64;
        total_time += dt;
        if delta > 0.0 {
            gains += delta * dt;
        } else {
            losses -= delta * dt;
        }
    }

    if total_time == 0.0 {
        return 0.5;
    }
    let avg_gain = gains / total_time;
    let avg_loss = losses / total_time;

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
            days_to_expiry: 15.0,
            is_crypto: 1.0,
            price_change_1d: 0.03,
            price_change_1w: -0.05,
            days_since_created: 20.0,
            created_to_expiry_span: 30.0,
            is_sports: 0.0,
            q_length: 40.0,
            q_word_count: 8.0,
            q_avg_word_len: 5.0,
            q_word_diversity: 1.0,
            q_has_number: 1.0,
            q_has_year: 1.0,
            q_has_percent: 0.0,
            q_has_dollar: 0.0,
            q_has_date: 0.0,
            q_starts_will: 1.0,
            q_has_by: 0.0,
            q_has_before: 0.0,
            q_has_above: 0.0,
            q_sentiment_pos: 0.0,
            q_sentiment_neg: 0.0,
            q_certainty: 1.0,
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
