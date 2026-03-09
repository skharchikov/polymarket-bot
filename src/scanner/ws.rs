use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tokio_tungstenite::tungstenite::Message;

const WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const PING_INTERVAL: Duration = Duration::from_secs(10);
const RECONNECT_DELAY: Duration = Duration::from_secs(5);

/// Max token IDs per subscription (API limit / performance).
const MAX_SUBSCRIPTIONS: usize = 200;

/// An activity alert from the websocket — a market had a significant price move.
#[derive(Debug, Clone)]
pub struct ActivityAlert {
    /// The YES token asset ID that moved.
    pub asset_id: String,
    /// Current best bid price (0-1).
    pub price: f64,
    /// Previous price we tracked (0-1).
    pub prev_price: f64,
    /// Trade size in USD (if triggered by a trade).
    pub trade_size: Option<f64>,
}

/// Tracks price state per asset for detecting significant moves.
struct PriceTracker {
    prices: HashMap<String, f64>,
    min_delta: f64,
    min_trade_usd: f64,
}

impl PriceTracker {
    fn new(min_delta: f64, min_trade_usd: f64) -> Self {
        Self {
            prices: HashMap::new(),
            min_delta,
            min_trade_usd,
        }
    }

    fn update_price(&mut self, asset_id: &str, new_price: f64) -> Option<ActivityAlert> {
        let prev = self.prices.get(asset_id).copied();
        self.prices.insert(asset_id.to_string(), new_price);

        let prev_price = prev?;
        let delta = (new_price - prev_price).abs();
        if delta >= self.min_delta {
            Some(ActivityAlert {
                asset_id: asset_id.to_string(),
                price: new_price,
                prev_price,
                trade_size: None,
            })
        } else {
            None
        }
    }

    fn check_trade(&mut self, asset_id: &str, price: f64, size: f64) -> Option<ActivityAlert> {
        let trade_usd = price * size;
        let prev_price = self.prices.get(asset_id).copied().unwrap_or(price);
        self.prices.insert(asset_id.to_string(), price);

        if trade_usd >= self.min_trade_usd {
            Some(ActivityAlert {
                asset_id: asset_id.to_string(),
                price,
                prev_price,
                trade_size: Some(trade_usd),
            })
        } else {
            None
        }
    }
}

/// Manages a persistent WebSocket connection to Polymarket's market channel.
/// Detects significant price moves and large trades, sending alerts through
/// an mpsc channel for the scanner to pick up.
pub struct MarketWatcher {
    alert_tx: mpsc::Sender<ActivityAlert>,
    /// Shared token list — updated externally by the refresh loop.
    pub tokens: Arc<RwLock<Vec<String>>>,
    min_price_delta: f64,
    min_trade_usd: f64,
}

impl MarketWatcher {
    pub fn new(
        alert_tx: mpsc::Sender<ActivityAlert>,
        min_price_delta: f64,
        min_trade_usd: f64,
    ) -> Self {
        Self {
            alert_tx,
            tokens: Arc::new(RwLock::new(Vec::new())),
            min_price_delta,
            min_trade_usd,
        }
    }

    /// Run forever: connect, subscribe, process messages, reconnect on failure.
    pub async fn run(&self) -> ! {
        loop {
            let token_ids = self.tokens.read().await.clone();
            if token_ids.is_empty() {
                tracing::debug!("No tokens to watch, waiting...");
                tokio::time::sleep(Duration::from_secs(30)).await;
                continue;
            }

            tracing::info!(
                tokens = token_ids.len(),
                "Connecting to Polymarket WebSocket..."
            );

            match self.connect_and_stream(&token_ids).await {
                Ok(()) => {
                    tracing::info!("WebSocket stream ended, reconnecting...");
                }
                Err(e) => {
                    tracing::warn!(err = %e, "WebSocket connection failed, reconnecting...");
                }
            }

            tokio::time::sleep(RECONNECT_DELAY).await;
        }
    }

    async fn connect_and_stream(&self, token_ids: &[String]) -> Result<()> {
        let (ws_stream, _) = tokio_tungstenite::connect_async(WS_URL)
            .await
            .context("WebSocket connect failed")?;

        let (mut write, mut read) = ws_stream.split();

        // Subscribe — limit to MAX_SUBSCRIPTIONS
        let ids: Vec<&String> = token_ids.iter().take(MAX_SUBSCRIPTIONS).collect();
        let sub_msg = serde_json::json!({
            "assets_ids": ids,
            "type": "market",
            "custom_feature_enabled": true,
        });
        write
            .send(Message::Text(sub_msg.to_string().into()))
            .await
            .context("Failed to send subscription")?;

        tracing::info!(tokens = ids.len(), "Subscribed to market channel");

        let mut tracker = PriceTracker::new(self.min_price_delta, self.min_trade_usd);
        let mut last_ping = Instant::now();

        loop {
            if last_ping.elapsed() >= PING_INTERVAL {
                write
                    .send(Message::Text("PING".into()))
                    .await
                    .context("Failed to send PING")?;
                last_ping = Instant::now();
            }

            let msg = tokio::time::timeout(PING_INTERVAL, read.next()).await;

            let msg = match msg {
                Ok(Some(Ok(msg))) => msg,
                Ok(Some(Err(e))) => {
                    anyhow::bail!("WebSocket read error: {e}");
                }
                Ok(None) => {
                    anyhow::bail!("WebSocket stream ended");
                }
                Err(_) => continue,
            };

            match msg {
                Message::Text(text) => {
                    let text = text.as_ref();
                    if text == "PONG" {
                        continue;
                    }
                    self.handle_message(text, &mut tracker).await;
                }
                Message::Close(_) => {
                    tracing::info!("WebSocket server closed connection");
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    async fn handle_message(&self, text: &str, tracker: &mut PriceTracker) {
        let json: serde_json::Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(_) => return,
        };

        let event_type = json["event_type"].as_str().unwrap_or("");

        match event_type {
            "best_bid_ask" => {
                let asset_id = match json["asset_id"].as_str() {
                    Some(id) => id,
                    None => return,
                };
                let best_bid = json["best_bid"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                if let Some(alert) = tracker.update_price(asset_id, best_bid) {
                    tracing::info!(
                        asset = &alert.asset_id[..16.min(alert.asset_id.len())],
                        price = format_args!("{:.1}%", alert.price * 100.0),
                        delta = format_args!("{:+.1}%", (alert.price - alert.prev_price) * 100.0),
                        "Price move detected"
                    );
                    let _ = self.alert_tx.try_send(alert);
                }
            }

            "price_change" => {
                if let Some(changes) = json["price_changes"].as_array() {
                    for change in changes {
                        let asset_id = match change["asset_id"].as_str() {
                            Some(id) => id,
                            None => continue,
                        };
                        let best_bid = change["best_bid"]
                            .as_str()
                            .and_then(|s| s.parse::<f64>().ok());
                        if let Some(price) = best_bid
                            && let Some(alert) = tracker.update_price(asset_id, price)
                        {
                            tracing::info!(
                                asset = &alert.asset_id[..16.min(alert.asset_id.len())],
                                delta = format_args!(
                                    "{:+.1}%",
                                    (alert.price - alert.prev_price) * 100.0
                                ),
                                "Price change alert"
                            );
                            let _ = self.alert_tx.try_send(alert);
                        }
                    }
                }
            }

            "last_trade_price" => {
                let asset_id = match json["asset_id"].as_str() {
                    Some(id) => id,
                    None => return,
                };
                let price = json["price"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let size = json["size"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                if let Some(alert) = tracker.check_trade(asset_id, price, size) {
                    tracing::info!(
                        asset = &alert.asset_id[..16.min(alert.asset_id.len())],
                        trade_usd = format_args!("${:.0}", alert.trade_size.unwrap_or(0.0)),
                        price = format_args!("{:.1}%", alert.price * 100.0),
                        "Large trade detected"
                    );
                    let _ = self.alert_tx.try_send(alert);
                }
            }

            "market_resolved" => {
                let market = json["market"].as_str().unwrap_or("?");
                let outcome = json["winning_outcome"].as_str().unwrap_or("?");
                tracing::info!(
                    market = &market[..16.min(market.len())],
                    outcome,
                    "Market resolved via WS"
                );
            }

            _ => {}
        }
    }
}
