use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single price tick from CLOB history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceTick {
    /// Unix timestamp
    pub t: i64,
    /// Price (0.0 - 1.0)
    pub p: f64,
}

/// Response from the CLOB prices-history endpoint.
#[derive(Debug, Deserialize)]
pub struct PriceHistoryResponse {
    pub history: Vec<PriceTick>,
}

/// A fully resolved market with its history, ready for backtesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalMarket {
    pub market_id: String,
    pub question: String,
    pub token_id: String,
    pub end_date: DateTime<Utc>,
    /// true = YES won, false = NO won
    pub resolved_yes: bool,
    pub price_history: Vec<PriceTick>,
}

impl HistoricalMarket {
    /// Duration of the market in hours based on first and last price tick.
    pub fn duration_hours(&self) -> f64 {
        if self.price_history.len() < 2 {
            return 0.0;
        }
        let first = self.price_history.first().unwrap().t;
        let last = self.price_history.last().unwrap().t;
        (last - first) as f64 / 3600.0
    }
}

/// Compact market info from Gamma API for crawling.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GammaMarket {
    #[serde(rename = "id")]
    pub market_id: String,
    pub question: String,
    #[serde(default)]
    pub clob_token_ids: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(default)]
    pub outcome_prices: Option<String>,
    #[serde(default)]
    pub outcomes: Option<String>,
    #[serde(default)]
    pub slug: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub volume_num: f64,
    #[serde(default)]
    pub liquidity_num: f64,
    #[serde(default)]
    pub one_day_price_change: Option<f64>,
    #[serde(default)]
    pub one_week_price_change: Option<f64>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub events: Vec<GammaEvent>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GammaEvent {
    pub slug: String,
}

impl GammaMarket {
    pub fn event_slug(&self) -> Option<&str> {
        self.events.first().map(|e| e.slug.as_str())
    }

    pub fn resolved_yes(&self) -> Option<bool> {
        let prices_str = self.outcome_prices.as_ref()?;
        let prices: Vec<String> = serde_json::from_str(prices_str).ok()?;
        let yes_price: f64 = prices.first()?.parse().ok()?;
        let no_price: f64 = prices.get(1)?.parse().ok()?;

        if yes_price == 1.0 && no_price == 0.0 {
            Some(true)
        } else if yes_price == 0.0 && no_price == 1.0 {
            Some(false)
        } else {
            None
        }
    }

    pub fn yes_token_id(&self) -> Option<String> {
        let ids_str = self.clob_token_ids.as_ref()?;
        let ids: Vec<String> = serde_json::from_str(ids_str).ok()?;
        ids.into_iter().next().filter(|s| !s.is_empty())
    }

    /// True if this is a binary YES/NO market (exactly 2 outcomes).
    /// Multi-outcome markets (e.g. "Who will win?" with 5 candidates) return false.
    pub fn is_binary(&self) -> bool {
        // Check outcomes field: should be ["Yes","No"]
        if let Some(outcomes_str) = &self.outcomes
            && let Ok(outcomes) = serde_json::from_str::<Vec<String>>(outcomes_str)
        {
            return outcomes.len() == 2;
        }
        // Fallback: check token IDs count (binary markets have exactly 2)
        if let Some(ids_str) = &self.clob_token_ids
            && let Ok(ids) = serde_json::from_str::<Vec<String>>(ids_str)
        {
            return ids.len() == 2;
        }
        false
    }

    pub fn polymarket_url(&self) -> String {
        let event_slug = self.events.first().map(|e| e.slug.as_str());
        let market_slug = self.slug.as_deref();
        match event_slug {
            Some(ev) => match market_slug {
                Some(mk) if mk != ev => format!("https://polymarket.com/event/{ev}/{mk}"),
                _ => format!("https://polymarket.com/event/{ev}"),
            },
            None => match market_slug {
                Some(slug) => format!("https://polymarket.com/event/{slug}"),
                None => format!("https://polymarket.com/event/{}", self.market_id),
            },
        }
    }

    pub fn is_crypto_related(&self) -> bool {
        let q = self.question.to_lowercase();
        let cat = self.category.as_deref().unwrap_or_default().to_lowercase();

        const KEYWORDS: &[&str] = &[
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "solana",
            "sol",
            "crypto",
            "defi",
            "nft",
            "blockchain",
            "dogecoin",
            "doge",
            "xrp",
            "ripple",
            "cardano",
            "polkadot",
            "avalanche",
            "chainlink",
            "bnb",
            "binance",
            "coinbase",
            "stablecoin",
            "memecoin",
        ];

        KEYWORDS.iter().any(|kw| q.contains(kw)) || cat.contains("crypto")
    }

    /// Returns true for short-duration noise markets (5-min up/down, etc).
    pub fn is_short_duration_noise(&self) -> bool {
        let q = self.question.to_lowercase();
        q.contains("up or down")
    }

    /// Extract the YES outcome price from `outcomePrices` / `outcome_prices`.
    pub fn yes_price(&self) -> Option<f64> {
        let s = self.outcome_prices.as_ref()?;
        let p: Vec<String> = serde_json::from_str(s).ok()?;
        p.first().and_then(|v| v.parse::<f64>().ok())
    }
}

const GAMMA_API: &str = "https://gamma-api.polymarket.com";

/// Fetch current YES prices for a batch of market IDs concurrently.
///
/// Returns a `Vec<Option<f64>>` aligned with the input slice.
pub async fn fetch_yes_prices(http: &reqwest::Client, market_ids: &[&str]) -> Vec<Option<f64>> {
    let futs: Vec<_> = market_ids
        .iter()
        .map(|id| {
            let url = format!("{GAMMA_API}/markets/{id}");
            async move {
                let resp = http.get(&url).send().await.ok()?;
                let text = resp.text().await.ok()?;
                let market: GammaMarket = serde_json::from_str(&text).ok()?;
                market.yes_price()
            }
        })
        .collect();
    futures_util::future::join_all(futs).await
}
