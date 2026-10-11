//! Best-effort current YES prices with a short TTL cache. Never errors:
//! missing prices render as "—".

use std::collections::HashMap;
use std::time::{Duration, Instant};

use polymarket_common::data::models::GammaMarket;
use tokio::sync::Mutex;

pub const GAMMA_API: &str = "https://gamma-api.polymarket.com";

pub struct PriceCache {
    http: reqwest::Client,
    ttl: Duration,
    base: String,
    map: Mutex<HashMap<String, (Instant, f64)>>,
}

pub fn stale_ids(
    map: &HashMap<String, (Instant, f64)>,
    ids: &[String],
    now: Instant,
    ttl: Duration,
) -> Vec<String> {
    ids.iter()
        .filter(|id| {
            map.get(*id)
                .is_none_or(|(t, _)| now.duration_since(*t) > ttl)
        })
        .cloned()
        .collect()
}

impl PriceCache {
    pub fn new(http: reqwest::Client, ttl: Duration) -> Self {
        Self::with_base(http, ttl, GAMMA_API.into())
    }

    pub fn with_base(http: reqwest::Client, ttl: Duration, base: String) -> Self {
        Self {
            http,
            ttl,
            base,
            map: Mutex::new(HashMap::new()),
        }
    }

    async fn fetch_one(&self, id: &str) -> Option<f64> {
        let m: GammaMarket = self
            .http
            .get(format!("{}/markets/{id}", self.base))
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json()
            .await
            .ok()?;
        m.yes_price()
    }

    pub async fn yes_prices(&self, ids: &[String]) -> HashMap<String, f64> {
        let now = Instant::now();
        let todo = {
            let map = self.map.lock().await;
            stale_ids(&map, ids, now, self.ttl)
        };
        let fetched =
            futures_util::future::join_all(todo.iter().map(|id| self.fetch_one(id))).await;
        let mut map = self.map.lock().await;
        for (id, p) in todo.into_iter().zip(fetched) {
            if let Some(p) = p {
                map.insert(id, (now, p));
            }
        }
        ids.iter()
            .filter_map(|id| map.get(id).map(|(_, p)| (id.clone(), *p)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_ids_returns_missing_and_expired_only() {
        let now = Instant::now();
        let mut map = HashMap::new();
        map.insert("fresh".to_string(), (now, 0.5));
        map.insert("old".to_string(), (now - Duration::from_secs(120), 0.5));
        let ids = vec!["fresh".to_string(), "old".to_string(), "new".to_string()];
        let mut stale = stale_ids(&map, &ids, now, Duration::from_secs(60));
        stale.sort();
        assert_eq!(stale, vec!["new".to_string(), "old".to_string()]);
    }

    #[tokio::test]
    async fn unreachable_gamma_yields_empty_map_quickly() {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(300))
            .build()
            .unwrap();
        let cache =
            PriceCache::with_base(http, Duration::from_secs(60), "http://127.0.0.1:9".into());
        let t = Instant::now();
        let got = cache.yes_prices(&["m1".to_string()]).await;
        assert!(got.is_empty());
        assert!(t.elapsed() < Duration::from_secs(2));
    }
}
