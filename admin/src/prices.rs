//! Best-effort current YES prices with a short TTL cache. Never errors:
//! missing prices render as "—".

use std::collections::HashMap;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use polymarket_common::data::models::GammaMarket;
use tokio::sync::Mutex;

/// Upper bound for one whole fetch fan-out; cached values are used past it.
const FETCH_BUDGET: Duration = Duration::from_secs(3);
/// How long a failed id is not retried.
const MISS_TTL: Duration = Duration::from_secs(30);
const CONCURRENCY: usize = 8;

pub const GAMMA_API: &str = "https://gamma-api.polymarket.com";

pub struct PriceCache {
    http: reqwest::Client,
    ttl: Duration,
    base: String,
    map: Mutex<HashMap<String, (Instant, f64)>>,
    misses: Mutex<HashMap<String, Instant>>,
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
            misses: Mutex::new(HashMap::new()),
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
        let mut seen = std::collections::HashSet::new();
        let uniq: Vec<String> = ids.iter().filter(|id| seen.insert(*id)).cloned().collect();
        let stale = {
            let map = self.map.lock().await;
            stale_ids(&map, &uniq, now, self.ttl)
        };
        let todo: Vec<String> = {
            let misses = self.misses.lock().await;
            stale
                .into_iter()
                .filter(|id| {
                    misses
                        .get(id)
                        .is_none_or(|t| now.duration_since(*t) > MISS_TTL)
                })
                .collect()
        };
        if !todo.is_empty() {
            let deadline = tokio::time::Instant::now() + FETCH_BUDGET;
            let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut stream = futures_util::stream::iter(todo.iter().cloned())
                .map(|id| async move {
                    let p = self.fetch_one(&id).await;
                    (id, p)
                })
                .buffer_unordered(CONCURRENCY);
            // Record each result as it arrives so a slow straggler cannot discard
            // the ones that already finished.
            while let Ok(Some((id, p))) = tokio::time::timeout_at(deadline, stream.next()).await {
                let mut map = self.map.lock().await;
                let mut misses = self.misses.lock().await;
                match p {
                    Some(p) => {
                        misses.remove(&id);
                        map.insert(id.clone(), (now, p));
                    }
                    None => {
                        misses.insert(id.clone(), now);
                    }
                }
                done.insert(id);
            }
            // Budget exhausted: negative-cache only what did not finish.
            let mut misses = self.misses.lock().await;
            for id in todo.iter().filter(|id| !done.contains(*id)) {
                misses.insert(id.clone(), now);
            }
        }
        let map = self.map.lock().await;
        uniq.iter()
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

    #[tokio::test]
    async fn failed_ids_are_negative_cached_and_deduped() {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(300))
            .build()
            .unwrap();
        let cache =
            PriceCache::with_base(http, Duration::from_secs(60), "http://127.0.0.1:9".into());
        let ids = vec!["m1".to_string(), "m1".to_string()];
        assert!(cache.yes_prices(&ids).await.is_empty());
        assert_eq!(cache.misses.lock().await.len(), 1);
        let t = Instant::now();
        assert!(cache.yes_prices(&ids).await.is_empty());
        assert!(t.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn slow_gamma_is_bounded_by_budget() {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((s, _)) = l.accept().await {
                held.push(s); // accept, never answer
            }
        });
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        let cache = PriceCache::with_base(http, Duration::from_secs(60), format!("http://{addr}"));
        let t = Instant::now();
        assert!(cache.yes_prices(&["m1".to_string()]).await.is_empty());
        assert!(t.elapsed() < Duration::from_secs(4));
        let t = Instant::now();
        cache.yes_prices(&["m1".to_string()]).await;
        assert!(t.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn fast_ids_survive_when_another_id_hangs() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut s, _)) = l.accept().await {
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 2048];
                    let n = s.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    if req.contains("/markets/fast") {
                        let body =
                            r#"{"id":"fast","question":"q","outcomePrices":"[\"0.42\",\"0.58\"]"}"#;
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        );
                        let _ = s.write_all(resp.as_bytes()).await;
                    } else {
                        tokio::time::sleep(Duration::from_secs(60)).await;
                    }
                });
            }
        });
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        let cache = PriceCache::with_base(http, Duration::from_secs(60), format!("http://{addr}"));
        let ids = vec!["fast".to_string(), "hang".to_string()];
        let got = cache.yes_prices(&ids).await;
        assert_eq!(got.get("fast"), Some(&0.42));
        assert!(!got.contains_key("hang"));
        assert!(cache.misses.lock().await.contains_key("hang"));
        assert!(!cache.misses.lock().await.contains_key("fast"));
        // Cached: second call returns promptly with the fast price.
        let t = Instant::now();
        assert_eq!(cache.yes_prices(&ids).await.get("fast"), Some(&0.42));
        assert!(t.elapsed() < Duration::from_millis(50));
    }
}
