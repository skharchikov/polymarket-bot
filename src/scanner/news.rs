use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use std::time::Duration;

use crate::data::models::GammaMarket;

/// A news item from any source.
#[derive(Debug, Clone)]
pub struct NewsItem {
    pub title: String,
    pub source: String,
    #[allow(dead_code)]
    pub url: String,
    pub published: Option<DateTime<Utc>>,
    pub summary: String,
}

/// A market matched with relevant news — the core of our edge.
#[derive(Debug, Clone)]
pub struct NewsMatch {
    pub market: GammaMarket,
    pub news: Vec<NewsItem>,
    pub relevance_score: f64,
}

/// Minimum cosine similarity to consider a news item relevant to a market.
const EMBEDDING_MIN_SIMILARITY: f64 = 0.35;

pub struct NewsAggregator {
    http: Client,
    openai_api_key: Option<String>,
}

impl NewsAggregator {
    pub fn new(http: Client) -> Self {
        let openai_api_key = std::env::var("OPENAI_API_KEY").ok();
        Self {
            http,
            openai_api_key,
        }
    }

    /// Fetch recent news from all free sources.
    /// Returns (news_items, per_source_counts).
    pub async fn fetch_all(&self) -> (Vec<NewsItem>, Vec<(String, usize)>) {
        let mut all = Vec::new();
        let mut source_counts = Vec::new();

        // Parallel fetch from multiple sources
        let (google, reddit, crypto, reuters) = tokio::join!(
            self.google_news_top(),
            self.reddit_news(),
            self.crypto_news(),
            self.reuters_news(),
        );

        for (name, result) in [
            ("Google News", google),
            ("Reddit", reddit),
            ("Crypto", crypto),
            ("Reuters", reuters),
        ] {
            match result {
                Ok(items) => {
                    tracing::info!(source = name, count = items.len(), "Fetched news");
                    source_counts.push((name.to_string(), items.len()));
                    all.extend(items);
                }
                Err(e) => {
                    tracing::warn!(source = name, err = %e, "News fetch failed");
                    source_counts.push((name.to_string(), 0));
                }
            }
        }

        // Deduplicate by title similarity
        dedup_news(&mut all);

        tracing::info!(count = all.len(), "News items fetched");
        (all, source_counts)
    }

    /// Google News RSS — top stories. Free, no key.
    async fn google_news_top(&self) -> Result<Vec<NewsItem>> {
        let mut items = Vec::new();

        // Top stories + specific searches for common Polymarket topics
        let feeds = [
            "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=politics+election&hl=en-US",
            "https://news.google.com/rss/search?q=crypto+bitcoin+regulation&hl=en-US",
            "https://news.google.com/rss/search?q=AI+technology+regulation&hl=en-US",
            "https://news.google.com/rss/search?q=federal+reserve+interest+rates&hl=en-US",
        ];

        for feed_url in feeds {
            match self.fetch_rss(feed_url, "Google News").await {
                Ok(feed_items) => items.extend(feed_items),
                Err(e) => tracing::debug!(url = feed_url, err = %e, "RSS fetch failed"),
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(items)
    }

    /// Fetch an RSS feed with retry. Source tag is applied to all items.
    async fn fetch_rss(&self, url: &str, source: &str) -> Result<Vec<NewsItem>> {
        let text = fetch_with_retry(&self.http, url, "Mozilla/5.0").await?;

        let mut items = Vec::new();
        for item_block in text.split("<item>").skip(1) {
            let title = extract_tag(item_block, "title").unwrap_or_default();
            let link = extract_tag(item_block, "link").unwrap_or_default();
            let pub_date = extract_tag(item_block, "pubDate");
            let description = extract_tag(item_block, "description").unwrap_or_default();

            // Clean HTML from description
            let clean_desc = strip_html(&description);

            let published = pub_date.and_then(|d| parse_rss_date(&d));

            if !title.is_empty() {
                items.push(NewsItem {
                    title,
                    source: source.to_string(),
                    url: link,
                    published,
                    summary: truncate(&clean_desc, 200),
                });
            }
        }

        Ok(items)
    }

    /// Reddit content via Google News search (Reddit blocks server IPs directly).
    async fn reddit_news(&self) -> Result<Vec<NewsItem>> {
        let mut items = Vec::new();
        let queries = [
            "site:reddit.com+news",
            "site:reddit.com+politics",
            "site:reddit.com+crypto",
        ];

        for query in &queries {
            let url =
                format!("https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en");
            match self.fetch_rss(&url, "Reddit").await {
                Ok(feed_items) => items.extend(feed_items),
                Err(e) => tracing::debug!(query = query, err = %e, "Reddit via Google failed"),
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(items)
    }

    /// Crypto news from multiple RSS feeds. Free, no key.
    async fn crypto_news(&self) -> Result<Vec<NewsItem>> {
        let feeds = [
            (
                "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "CoinDesk",
            ),
            ("https://cointelegraph.com/rss", "CoinTelegraph"),
            ("https://decrypt.co/feed", "Decrypt"),
            ("https://blockworks.co/feed", "Blockworks"),
            ("https://www.theblock.co/rss.xml", "The Block"),
            ("https://bitcoinmagazine.com/feed", "Bitcoin Magazine"),
            ("https://www.dlnews.com/arc/outboundfeeds/rss/", "DL News"),
        ];

        let mut items = Vec::new();
        for (url, source) in feeds {
            match self.fetch_rss(url, source).await {
                Ok(feed_items) => items.extend(feed_items),
                Err(e) => tracing::debug!(source = source, err = %e, "Crypto RSS failed"),
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(items)
    }

    /// Reuters + AP + BBC via Google News RSS proxies.
    async fn reuters_news(&self) -> Result<Vec<NewsItem>> {
        let feeds = [
            "https://news.google.com/rss/search?q=site:reuters.com&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=site:apnews.com&hl=en-US&gl=US&ceid=US:en",
            "https://feeds.bbci.co.uk/news/world/rss.xml",
        ];

        let mut items = Vec::new();
        for url in feeds {
            match self.fetch_rss(url, "Reuters").await {
                Ok(feed_items) => items.extend(feed_items),
                Err(e) => tracing::debug!(url = url, err = %e, "Reuters RSS failed"),
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(items)
    }

    /// Match news to markets using OpenAI embeddings (cosine similarity).
    pub async fn match_to_markets(
        &self,
        news: &[NewsItem],
        markets: &[GammaMarket],
    ) -> Result<Vec<NewsMatch>> {
        let api_key = self
            .openai_api_key
            .as_deref()
            .context("OPENAI_API_KEY required for embedding-based news matching")?;
        if news.is_empty() || markets.is_empty() {
            return Ok(Vec::new());
        }

        // Build all texts to embed in a single batch
        let news_texts: Vec<String> = news.iter().map(|n| n.title.clone()).collect();
        let market_texts: Vec<String> = markets.iter().map(|m| m.question.clone()).collect();

        let mut all_texts = news_texts.clone();
        all_texts.extend(market_texts.clone());

        let embeddings = self.embed_texts(&all_texts, api_key).await?;

        let news_embeddings = &embeddings[..news.len()];
        let market_embeddings = &embeddings[news.len()..];

        let mut matches: Vec<NewsMatch> = Vec::new();

        for (mi, market) in markets.iter().enumerate() {
            let market_emb = &market_embeddings[mi];

            let mut scored_news: Vec<(f64, &NewsItem)> = news
                .iter()
                .enumerate()
                .map(|(ni, item)| {
                    let sim = cosine_similarity(market_emb, &news_embeddings[ni]);
                    (sim, item)
                })
                .filter(|(sim, _)| *sim > EMBEDDING_MIN_SIMILARITY)
                .collect();

            scored_news.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            if !scored_news.is_empty() {
                let best_score = scored_news[0].0;
                let matched_news: Vec<NewsItem> = scored_news
                    .iter()
                    .take(5)
                    .map(|(_, item)| (*item).clone())
                    .collect();

                matches.push(NewsMatch {
                    market: market.clone(),
                    news: matched_news,
                    relevance_score: best_score,
                });
            }
        }

        matches.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        tracing::info!(
            matched = matches.len(),
            total_markets = markets.len(),
            total_news = news.len(),
            "Embedding-based matching complete"
        );

        Ok(matches)
    }

    /// Call OpenAI embeddings API. Returns one vector per input text.
    async fn embed_texts(&self, texts: &[String], api_key: &str) -> Result<Vec<Vec<f64>>> {
        let resp = self
            .http
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&serde_json::json!({
                "model": "text-embedding-3-small",
                "input": texts,
            }))
            .send()
            .await
            .context("embeddings request failed")?;

        if !resp.status().is_success() {
            let body: String = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI embeddings API error: {body}");
        }

        let json: serde_json::Value = resp.json().await.context("parsing embeddings response")?;

        let empty_vec = vec![];
        let mut embeddings: Vec<(usize, Vec<f64>)> = json["data"]
            .as_array()
            .context("no data array in response")?
            .iter()
            .map(|item| {
                let idx = item["index"].as_u64().unwrap_or(0) as usize;
                let emb: Vec<f64> = item["embedding"]
                    .as_array()
                    .unwrap_or(&empty_vec)
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .collect();
                (idx, emb)
            })
            .collect();

        // Sort by index to match input order
        embeddings.sort_by_key(|(idx, _)| *idx);
        Ok(embeddings.into_iter().map(|(_, emb)| emb).collect())
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// --- HTTP helpers ---

/// Fetch URL with up to 3 retries and exponential backoff.
async fn fetch_with_retry(http: &Client, url: &str, user_agent: &str) -> Result<String> {
    let mut last_err = None;

    for attempt in 0..3 {
        if attempt > 0 {
            let delay = Duration::from_secs(2u64.pow(attempt));
            tokio::time::sleep(delay).await;
        }

        match http
            .get(url)
            .header("User-Agent", user_agent)
            .timeout(Duration::from_secs(15))
            .send()
            .await
        {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return resp
                        .text()
                        .await
                        .map_err(|e| anyhow::anyhow!("reading body: {e}"));
                }
                let body = resp.text().await.unwrap_or_default();
                let preview = &body[..body.len().min(200)];
                tracing::warn!(
                    url = url,
                    status = %status,
                    attempt = attempt + 1,
                    body_prefix = preview,
                    "HTTP non-200, retrying"
                );
                last_err = Some(anyhow::anyhow!("HTTP {status}"));
            }
            Err(e) => {
                tracing::warn!(url = url, attempt = attempt + 1, err = %e, "HTTP error, retrying");
                last_err = Some(e.into());
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("fetch failed")))
}

// --- Text helpers ---

fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    let content = &xml[start..end];
    // Handle CDATA
    let content = if content.starts_with("<![CDATA[") {
        &content[9..content.len().saturating_sub(3)]
    } else {
        content
    };
    Some(content.trim().to_string())
}

fn strip_html(s: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .replace("&quot;", "\"")
}

fn parse_rss_date(s: &str) -> Option<DateTime<Utc>> {
    // RFC 2822 format: "Thu, 05 Mar 2026 19:30:00 GMT"
    DateTime::parse_from_rfc2822(s)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

fn dedup_news(items: &mut Vec<NewsItem>) {
    let mut seen = std::collections::HashSet::new();
    items.retain(|item| {
        let key = item
            .title
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .take(50)
            .collect::<String>();
        seen.insert(key)
    });
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn news(title: &str, summary: &str) -> NewsItem {
        NewsItem {
            title: title.to_string(),
            source: "test".to_string(),
            url: String::new(),
            published: None,
            summary: summary.to_string(),
        }
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10, "Identical vectors: {sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "Orthogonal vectors: {sim}");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10, "Opposite vectors: {sim}");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-10, "Zero vector: {sim}");
    }

    #[test]
    fn test_cosine_similarity_similar_direction() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; // same direction, different magnitude
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10, "Parallel vectors: {sim}");
    }

    #[test]
    fn test_dedup_news() {
        let mut items = vec![
            news("Breaking: Fed cuts rates", ""),
            news("BREAKING: Fed Cuts Rates!", ""),
            news("Different headline entirely", ""),
        ];
        dedup_news(&mut items);
        assert_eq!(items.len(), 2, "Near-duplicate should be removed");
    }

    #[test]
    fn test_dedup_preserves_distinct() {
        let mut items = vec![
            news("Bitcoin surges past 100k", ""),
            news("Ethereum hits new all-time high", ""),
            news("Federal Reserve cuts rates", ""),
        ];
        dedup_news(&mut items);
        assert_eq!(items.len(), 3, "All distinct — nothing removed");
    }

    #[test]
    fn test_embedding_min_similarity_threshold() {
        // Verify the threshold constant is in a reasonable range
        const { assert!(EMBEDDING_MIN_SIMILARITY > 0.0) };
        const { assert!(EMBEDDING_MIN_SIMILARITY < 1.0) };
    }
}
