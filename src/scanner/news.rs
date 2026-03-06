use anyhow::Result;
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

pub struct NewsAggregator {
    http: Client,
}

impl NewsAggregator {
    pub fn new(http: Client) -> Self {
        Self { http }
    }

    /// Fetch recent news from all free sources.
    pub async fn fetch_all(&self) -> Vec<NewsItem> {
        let mut all = Vec::new();

        // Parallel fetch from multiple sources
        let (google, reddit, polymarket_blog) = tokio::join!(
            self.google_news_top(),
            self.reddit_news(),
            self.polymarket_activity(),
        );

        if let Ok(items) = google {
            all.extend(items);
        }
        if let Ok(items) = reddit {
            all.extend(items);
        }
        if let Ok(items) = polymarket_blog {
            all.extend(items);
        }

        // Deduplicate by title similarity
        dedup_news(&mut all);

        tracing::info!(count = all.len(), "News items fetched");
        all
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
            match self.fetch_rss(feed_url).await {
                Ok(feed_items) => items.extend(feed_items),
                Err(e) => tracing::debug!(url = feed_url, err = %e, "RSS fetch failed"),
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(items)
    }

    /// Parse RSS XML (simple parser, no XML dep needed).
    async fn fetch_rss(&self, url: &str) -> Result<Vec<NewsItem>> {
        let resp = self
            .http
            .get(url)
            .header("User-Agent", "Mozilla/5.0")
            .send()
            .await?;
        let text = resp.text().await?;

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
                    source: "Google News".to_string(),
                    url: link,
                    published,
                    summary: truncate(&clean_desc, 200),
                });
            }
        }

        Ok(items)
    }

    /// Reddit hot posts from news-heavy subreddits. Free, no key.
    async fn reddit_news(&self) -> Result<Vec<NewsItem>> {
        let mut items = Vec::new();
        let subreddits = [
            "news",
            "worldnews",
            "politics",
            "technology",
            "CryptoCurrency",
        ];

        for sub in &subreddits {
            let url = format!("https://www.reddit.com/r/{sub}/hot.json?limit=10");
            match self.fetch_reddit(&url, sub).await {
                Ok(posts) => items.extend(posts),
                Err(e) => tracing::debug!(sub = sub, err = %e, "Reddit fetch failed"),
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        Ok(items)
    }

    async fn fetch_reddit(&self, url: &str, subreddit: &str) -> Result<Vec<NewsItem>> {
        let resp: serde_json::Value = self
            .http
            .get(url)
            .header("User-Agent", "polymarket-bot/0.1")
            .send()
            .await?
            .json()
            .await?;

        let mut items = Vec::new();
        if let Some(posts) = resp["data"]["children"].as_array() {
            for post in posts {
                let data = &post["data"];
                let title = data["title"].as_str().unwrap_or_default().to_string();
                let url = data["url"].as_str().unwrap_or_default().to_string();
                let created = data["created_utc"].as_f64().unwrap_or(0.0) as i64;
                let selftext = data["selftext"].as_str().unwrap_or_default();
                let score = data["score"].as_i64().unwrap_or(0);

                // Only high-engagement posts
                if score < 100 || title.is_empty() {
                    continue;
                }

                let published = DateTime::from_timestamp(created, 0);

                items.push(NewsItem {
                    title,
                    source: format!("r/{subreddit}"),
                    url,
                    published,
                    summary: truncate(selftext, 200),
                });
            }
        }

        Ok(items)
    }

    /// Polymarket's own API — check for recently active/trending markets.
    async fn polymarket_activity(&self) -> Result<Vec<NewsItem>> {
        // Use Gamma API to find markets with recent large volume spikes
        // These often indicate news-driven activity
        let url = "https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=20&order=volumeNum&ascending=false";
        let resp: Vec<serde_json::Value> = self.http.get(url).send().await?.json().await?;

        let mut items = Vec::new();
        for market in resp.iter().take(10) {
            let question = market["question"].as_str().unwrap_or_default();
            let volume = market["volumeNum"]
                .as_f64()
                .or_else(|| market["volumeNum"].as_str().and_then(|s| s.parse().ok()))
                .unwrap_or(0.0);

            if volume > 100_000.0 && !question.is_empty() {
                items.push(NewsItem {
                    title: format!("Polymarket trending: {question}"),
                    source: "Polymarket".to_string(),
                    url: String::new(),
                    published: Some(Utc::now()),
                    summary: format!("Volume: ${volume:.0}"),
                });
            }
        }

        Ok(items)
    }

    /// Match news items to markets by keyword overlap.
    /// Returns markets with matched news, sorted by relevance.
    pub fn match_to_markets(news: &[NewsItem], markets: &[GammaMarket]) -> Vec<NewsMatch> {
        let mut matches: Vec<NewsMatch> = Vec::new();

        for market in markets {
            let market_words = extract_keywords(&market.question);
            if market_words.is_empty() {
                continue;
            }

            let mut matched_news = Vec::new();
            let mut best_score = 0.0_f64;

            for item in news {
                let news_words = extract_keywords(&item.title);
                let summary_words = extract_keywords(&item.summary);

                // Count keyword overlap
                let title_overlap = market_words
                    .iter()
                    .filter(|w| news_words.contains(w))
                    .count();
                let summary_overlap = market_words
                    .iter()
                    .filter(|w| summary_words.contains(w))
                    .count();

                let overlap = title_overlap + summary_overlap / 2;

                // Need at least 2 keyword matches to consider relevant
                if overlap >= 2 {
                    let score = overlap as f64 / market_words.len().max(1) as f64;
                    best_score = best_score.max(score);
                    matched_news.push(item.clone());
                }
            }

            if !matched_news.is_empty() {
                // Keep top 5 most relevant news per market
                matched_news.truncate(5);
                matches.push(NewsMatch {
                    market: market.clone(),
                    news: matched_news,
                    relevance_score: best_score,
                });
            }
        }

        // Sort by relevance score (best matches first)
        matches.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        matches
    }
}

// --- Helpers ---

const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "in", "on", "at", "to", "by", "for", "of", "be", "is", "it", "or", "and",
    "with", "will", "this", "that", "has", "have", "had", "was", "were", "are", "been", "its",
    "not", "but", "from", "they", "their", "can", "do", "does", "did", "would", "could", "should",
    "may", "might", "shall", "about", "above", "below", "between", "before", "after", "than",
    "more", "most", "very", "just", "also", "how", "what", "when", "where", "who", "which", "if",
    "then", "so", "up", "out", "all", "over", "into", "new", "says", "said", "yes", "no", "price",
    "market", "markets",
];

fn extract_keywords(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .filter(|w| !STOP_WORDS.contains(w))
        .map(|w| w.to_string())
        .collect()
}

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
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
