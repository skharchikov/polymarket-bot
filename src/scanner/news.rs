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
    /// Returns (news_items, per_source_counts).
    pub async fn fetch_all(&self) -> (Vec<NewsItem>, Vec<(String, usize)>) {
        let mut all = Vec::new();
        let mut source_counts = Vec::new();

        // Parallel fetch from multiple sources
        let (google, reddit, polymarket_blog, coindesk, reuters) = tokio::join!(
            self.google_news_top(),
            self.reddit_news(),
            self.polymarket_activity(),
            self.coindesk_news(),
            self.reuters_news(),
        );

        for (name, result) in [
            ("Google News", google),
            ("Reddit", reddit),
            ("Polymarket", polymarket_blog),
            ("CoinDesk", coindesk),
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

    /// Reddit via RSS — no rate limits, no auth needed.
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
            let url = format!("https://www.reddit.com/r/{sub}/hot.rss?limit=15");
            match self.fetch_rss(&url, &format!("r/{sub}")).await {
                Ok(posts) => items.extend(posts),
                Err(e) => tracing::debug!(sub = sub, err = %e, "Reddit RSS failed"),
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        Ok(items)
    }

    /// CoinDesk RSS — crypto news. Free, no key.
    async fn coindesk_news(&self) -> Result<Vec<NewsItem>> {
        let feeds = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
        ];

        let mut items = Vec::new();
        for url in feeds {
            match self.fetch_rss(url, "CoinDesk").await {
                Ok(feed_items) => items.extend(feed_items),
                Err(e) => tracing::debug!(url = url, err = %e, "CoinDesk/CT RSS failed"),
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(items)
    }

    /// Reuters RSS — global breaking news. Free, no key.
    async fn reuters_news(&self) -> Result<Vec<NewsItem>> {
        let feeds = [
            "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
            "https://news.google.com/rss/search?q=site:reuters.com&hl=en-US",
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

    /// Match news items to markets using BM25 relevance scoring.
    /// Returns markets with matched news, sorted by relevance.
    pub fn match_to_markets(news: &[NewsItem], markets: &[GammaMarket]) -> Vec<NewsMatch> {
        let index = Bm25Index::build(news);

        let mut matches: Vec<NewsMatch> = Vec::new();

        for market in markets {
            let query_terms = extract_keywords(&market.question);
            if query_terms.is_empty() {
                continue;
            }

            let mut scored_news: Vec<(f64, &NewsItem)> = news
                .iter()
                .enumerate()
                .map(|(i, item)| (index.score(&query_terms, i), item))
                .filter(|(score, _)| *score > BM25_MIN_SCORE)
                .collect();

            // Sort by score descending
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

        // Sort by relevance score (best matches first)
        matches.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        matches
    }
}

// --- BM25 scoring ---

/// Minimum BM25 score to consider a news item relevant to a market.
const BM25_MIN_SCORE: f64 = 1.5;

/// BM25 parameters — standard values from the literature.
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// In-memory BM25 index over a set of news items.
struct Bm25Index {
    /// Tokenized documents: doc_idx → terms
    docs: Vec<Vec<String>>,
    /// Document frequency: term → number of docs containing it
    df: std::collections::HashMap<String, usize>,
    /// Average document length
    avgdl: f64,
    /// Total number of documents
    n: usize,
}

impl Bm25Index {
    /// Build an index from news items. Combines title (2× weight) + summary.
    fn build(news: &[NewsItem]) -> Self {
        let mut df: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut docs = Vec::with_capacity(news.len());
        let mut total_len = 0usize;

        for item in news {
            // Title terms appear twice for boosted weight
            let mut terms = extract_keywords(&item.title);
            terms.extend(extract_keywords(&item.title));
            terms.extend(extract_keywords(&item.summary));

            // Count unique terms for DF
            let unique: std::collections::HashSet<&str> =
                terms.iter().map(|s| s.as_str()).collect();
            for term in &unique {
                *df.entry((*term).to_string()).or_insert(0) += 1;
            }

            total_len += terms.len();
            docs.push(terms);
        }

        let n = docs.len();
        let avgdl = if n > 0 {
            total_len as f64 / n as f64
        } else {
            1.0
        };

        Self { docs, df, avgdl, n }
    }

    /// Compute BM25 score for a query against document at index `doc_idx`.
    fn score(&self, query_terms: &[String], doc_idx: usize) -> f64 {
        let doc = &self.docs[doc_idx];
        let dl = doc.len() as f64;
        let mut score = 0.0;

        for term in query_terms {
            // Term frequency in this document
            let tf = doc.iter().filter(|t| *t == term).count() as f64;
            if tf == 0.0 {
                continue;
            }

            // Inverse document frequency: log((N - df + 0.5) / (df + 0.5) + 1)
            let df = *self.df.get(term).unwrap_or(&0) as f64;
            let idf = ((self.n as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();

            // BM25 TF component
            let tf_norm =
                (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avgdl));

            score += idf * tf_norm;
        }

        score
    }
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
    use crate::data::models::GammaMarket;

    fn news(title: &str, summary: &str) -> NewsItem {
        NewsItem {
            title: title.to_string(),
            source: "test".to_string(),
            url: String::new(),
            published: None,
            summary: summary.to_string(),
        }
    }

    fn market(question: &str) -> GammaMarket {
        GammaMarket {
            market_id: "test".into(),
            question: question.into(),
            outcomes: None,
            outcome_prices: None,
            volume_num: 0.0,
            liquidity_num: 0.0,
            clob_token_ids: None,
            end_date: None,
            slug: None,
            category: None,
        }
    }

    #[test]
    fn test_bm25_exact_match_scores_high() {
        let items = vec![
            news("Federal Reserve cuts interest rates by 50 basis points", ""),
            news("New iPhone released by Apple", ""),
        ];
        let index = Bm25Index::build(&items);
        let query = extract_keywords("Will the Federal Reserve cut interest rates?");

        let score_fed = index.score(&query, 0);
        let score_iphone = index.score(&query, 1);

        assert!(
            score_fed > score_iphone,
            "Fed news should score higher: {score_fed} vs {score_iphone}"
        );
        assert!(
            score_fed > BM25_MIN_SCORE,
            "Fed news should pass threshold: {score_fed}"
        );
    }

    #[test]
    fn test_bm25_semantic_near_miss_keyword_overlap_would_miss() {
        // "rates" appears in both but old keyword overlap needed ≥2 exact matches
        let items = vec![news(
            "Fed cuts rates in surprise move",
            "Economy responds to monetary policy shift",
        )];
        let index = Bm25Index::build(&items);
        let query = extract_keywords("Will interest rates decrease this quarter?");

        let score = index.score(&query, 0);
        assert!(
            score > 0.0,
            "Should find partial match via 'rates': {score}"
        );
    }

    #[test]
    fn test_bm25_title_boost() {
        // Same word in title vs summary should score differently (title is 2x)
        let items = vec![
            news("Bitcoin crashes below 50k", "crypto market turmoil"),
            news(
                "Market update today",
                "Bitcoin crashes below 50k in crypto turmoil",
            ),
        ];
        let index = Bm25Index::build(&items);
        let query = extract_keywords("Will Bitcoin crash?");

        let score_title = index.score(&query, 0);
        let score_summary = index.score(&query, 1);

        assert!(
            score_title > score_summary,
            "Title match should score higher: {score_title} vs {score_summary}"
        );
    }

    #[test]
    fn test_bm25_idf_rewards_rare_terms() {
        // "quantum" is rare, "technology" is common → "quantum" match should contribute more
        let items = vec![
            news("Quantum computing breakthrough announced", ""),
            news("Technology stocks rise", ""),
            news("Technology companies report earnings", ""),
            news("New technology regulation proposed", ""),
        ];
        let index = Bm25Index::build(&items);
        let query = extract_keywords("quantum technology");

        let score_quantum = index.score(&query, 0);
        let score_tech1 = index.score(&query, 1);

        assert!(
            score_quantum > score_tech1,
            "Rare 'quantum' match should outscore common 'technology': {score_quantum} vs {score_tech1}"
        );
    }

    #[test]
    fn test_bm25_no_match_scores_zero() {
        let items = vec![news("Apple releases new MacBook", "")];
        let index = Bm25Index::build(&items);
        let query = extract_keywords("Will Bitcoin reach 100k?");

        let score = index.score(&query, 0);
        assert!(
            (score - 0.0).abs() < f64::EPSILON,
            "No overlap should score 0: {score}"
        );
    }

    #[test]
    fn test_bm25_empty_corpus() {
        let items: Vec<NewsItem> = vec![];
        let index = Bm25Index::build(&items);
        assert_eq!(index.n, 0);
        assert!((index.avgdl - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_match_to_markets_uses_bm25() {
        let items = vec![
            news(
                "Federal Reserve raises interest rates",
                "Inflation concerns drive policy",
            ),
            news("SpaceX launches Starship successfully", ""),
            news("Bitcoin ETF approved by SEC", "Crypto markets rally"),
        ];

        let markets = vec![
            market("Will the Federal Reserve raise interest rates in March?"),
            market("Will SpaceX launch Starship in Q1?"),
        ];

        let matches = NewsAggregator::match_to_markets(&items, &markets);

        assert!(!matches.is_empty(), "Should find matches");

        // Fed market should match Fed news
        let fed_match = matches
            .iter()
            .find(|m| m.market.question.contains("Federal"))
            .unwrap();
        assert!(fed_match.news.iter().any(|n| n.title.contains("Federal")));

        // SpaceX market should match SpaceX news
        let spacex_match = matches
            .iter()
            .find(|m| m.market.question.contains("SpaceX"))
            .unwrap();
        assert!(spacex_match.news.iter().any(|n| n.title.contains("SpaceX")));
    }

    #[test]
    fn test_match_to_markets_sorted_by_relevance() {
        let items = vec![
            news(
                "Bitcoin surges past 100k as ETF inflows accelerate",
                "Record institutional buying",
            ),
            news(
                "Bitcoin mining difficulty reaches all-time high",
                "Hash rate climbs",
            ),
            news("Weather forecast for tomorrow", ""),
        ];

        let markets = vec![
            market("Will Bitcoin reach 150k by end of year?"),
            market("Will it rain tomorrow in New York?"),
        ];

        let matches = NewsAggregator::match_to_markets(&items, &markets);

        // Bitcoin market should have higher relevance (2 matching articles)
        if matches.len() >= 2 {
            assert!(
                matches[0].relevance_score >= matches[1].relevance_score,
                "Should be sorted by relevance"
            );
        }
    }

    #[test]
    fn test_bm25_score_values_single_doc() {
        // Single doc, single query term → verify formula by hand.
        // Doc: "bitcoin" (title doubled → tf=2, dl=2)
        // N=1, df=1, avgdl=2
        // IDF = ln((1 - 1 + 0.5) / (1 + 0.5) + 1) = ln(4/3)
        // TF  = (2 * 2.2) / (2 + 1.2 * (1 - 0.75 + 0.75 * 2/2)) = 4.4 / 3.2
        // Score = ln(4/3) * 4.4/3.2
        let items = vec![news("bitcoin", "")];
        let index = Bm25Index::build(&items);
        let query = vec!["bitcoin".to_string()];

        let score = index.score(&query, 0);
        let expected = (4.0_f64 / 3.0).ln() * (4.4 / 3.2);
        assert!(
            (score - expected).abs() < 1e-10,
            "score {score} != expected {expected}"
        );
    }

    #[test]
    fn test_bm25_score_increases_with_more_query_terms() {
        // More matching query terms → higher score
        let items = vec![news(
            "Federal Reserve cuts interest rates",
            "Economy slows down",
        )];
        let index = Bm25Index::build(&items);

        let one_term = vec!["federal".to_string()];
        let two_terms = vec!["federal".to_string(), "reserve".to_string()];
        let three_terms = vec![
            "federal".to_string(),
            "reserve".to_string(),
            "rates".to_string(),
        ];

        let s1 = index.score(&one_term, 0);
        let s2 = index.score(&two_terms, 0);
        let s3 = index.score(&three_terms, 0);

        assert!(s2 > s1, "2 terms ({s2}) should score > 1 term ({s1})");
        assert!(s3 > s2, "3 terms ({s3}) should score > 2 terms ({s2})");
    }

    #[test]
    fn test_bm25_idf_values() {
        // Term in 1/10 docs should have higher IDF than term in 9/10 docs
        let mut items: Vec<NewsItem> = (0..9)
            .map(|i| news(&format!("common topic number {i}"), ""))
            .collect();
        items.push(news("rare unique unicorn", ""));

        let index = Bm25Index::build(&items);

        // "rare" appears in 1/10 docs, "common" in 9/10
        let df_rare = *index.df.get("rare").unwrap_or(&0) as f64;
        let df_common = *index.df.get("common").unwrap_or(&0) as f64;
        let n = index.n as f64;

        let idf_rare = ((n - df_rare + 0.5) / (df_rare + 0.5) + 1.0).ln();
        let idf_common = ((n - df_common + 0.5) / (df_common + 0.5) + 1.0).ln();

        assert!(
            idf_rare > idf_common,
            "rare IDF ({idf_rare}) should > common IDF ({idf_common})"
        );
        assert!(idf_rare > 1.0, "rare IDF should be significant: {idf_rare}");
        assert!(idf_common < 0.5, "common IDF should be low: {idf_common}");
    }

    #[test]
    fn test_bm25_score_threshold_calibration() {
        // Verify BM25_MIN_SCORE separates good from poor matches
        let items = vec![
            news("Federal Reserve raises interest rates", "Inflation data"),
            news("Cat video goes viral on TikTok", "Funny animals"),
        ];
        let index = Bm25Index::build(&items);
        let query = extract_keywords("Will Federal Reserve raise rates?");

        let good = index.score(&query, 0);
        let bad = index.score(&query, 1);

        assert!(
            good > BM25_MIN_SCORE,
            "Relevant match should exceed threshold: {good} > {BM25_MIN_SCORE}"
        );
        assert!(
            bad < BM25_MIN_SCORE,
            "Irrelevant match should be below threshold: {bad} < {BM25_MIN_SCORE}"
        );
    }

    #[test]
    fn test_extract_keywords_filters_stop_words() {
        let words = extract_keywords("Will the Federal Reserve cut interest rates?");
        assert!(words.contains(&"federal".to_string()));
        assert!(words.contains(&"reserve".to_string()));
        assert!(!words.contains(&"will".to_string()));
        assert!(!words.contains(&"the".to_string()));
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
}
