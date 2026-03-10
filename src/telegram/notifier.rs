use anyhow::{Context, Result};
use reqwest::Client;
use std::sync::atomic::{AtomicI64, Ordering};

pub struct TelegramNotifier {
    client: Client,
    bot_token: String,
    chat_id: String,
    /// Last processed update_id for polling
    last_update_id: AtomicI64,
}

impl TelegramNotifier {
    pub fn new(bot_token: &str, chat_id: &str) -> Self {
        Self {
            client: Client::new(),
            bot_token: bot_token.to_string(),
            chat_id: chat_id.to_string(),
            last_update_id: AtomicI64::new(0),
        }
    }

    pub async fn send(&self, message: &str) -> Result<()> {
        self.send_to(&self.chat_id, message).await
    }

    /// Send to owner + all subscribers (deduped).
    pub async fn broadcast(&self, subscribers: &[(String, Option<String>)], message: &str) {
        let recipients = 1 + subscribers
            .iter()
            .filter(|(id, _)| id != &self.chat_id)
            .count();
        let _ = self.send(message).await;
        for (id, username) in subscribers {
            if id != &self.chat_id {
                let label = username.as_deref().unwrap_or("unknown");
                if let Err(e) = self.send_to(id, message).await {
                    tracing::warn!(chat_id = %id, username = label, err = %e, "Failed to send to subscriber");
                }
            }
        }
        tracing::debug!(recipients = recipients, "Broadcast complete");
    }

    pub async fn send_to(&self, chat_id: &str, message: &str) -> Result<()> {
        self.send_to_with_mode(chat_id, message, "Markdown").await
    }

    async fn send_to_with_mode(
        &self,
        chat_id: &str,
        message: &str,
        parse_mode: &str,
    ) -> Result<()> {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.bot_token);

        let disclaimer = if parse_mode == "HTML" {
            "<i>This is not financial advice. Do your own research.</i>"
        } else {
            "_This is not financial advice. Do your own research._"
        };
        let full_message = format!("{message}\n\n{disclaimer}");

        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({
                "chat_id": chat_id,
                "text": full_message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": true,
            }))
            .send()
            .await
            .context("failed to send telegram message")?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(chat_id = chat_id, body = body, "Telegram send failed");
            anyhow::bail!("Telegram API error: {body}");
        }

        let preview = message.chars().take(60).collect::<String>();
        tracing::info!(chat_id = chat_id, len = message.len(), preview = %preview, "Telegram message sent");
        Ok(())
    }

    /// Send a GIF animation to a chat.
    pub async fn send_animation(&self, chat_id: &str, gif_url: &str) -> Result<()> {
        let url = format!(
            "https://api.telegram.org/bot{}/sendAnimation",
            self.bot_token
        );
        let form = reqwest::multipart::Form::new()
            .text("chat_id", chat_id.to_string())
            .text("animation", gif_url.to_string());
        let resp = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .context("failed to send telegram animation")?;
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(chat_id, body, "Telegram sendAnimation failed");
        }
        Ok(())
    }

    /// Send a GIF to owner + all subscribers.
    pub async fn broadcast_animation(
        &self,
        subscribers: &[(String, Option<String>)],
        gif_url: &str,
    ) {
        let _ = self.send_animation(&self.chat_id, gif_url).await;
        for (id, _) in subscribers {
            if id != &self.chat_id {
                let _ = self.send_animation(id, gif_url).await;
            }
        }
    }

    /// Poll for new commands. Returns (chat_id, command, username, first_name).
    pub async fn poll_commands(&self) -> Vec<(String, String, Option<String>, Option<String>)> {
        let offset = self.last_update_id.load(Ordering::Relaxed);
        let url = format!("https://api.telegram.org/bot{}/getUpdates", self.bot_token);

        let body = serde_json::json!({
            "offset": offset + 1,
            "timeout": 0,
            "allowed_updates": ["message"],
        });

        let resp = match self.client.post(&url).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(err = %e, "Telegram poll failed");
                return vec![];
            }
        };

        let json: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(_) => return vec![],
        };

        let mut commands = vec![];

        if let Some(updates) = json["result"].as_array() {
            for update in updates {
                if let Some(update_id) = update["update_id"].as_i64() {
                    self.last_update_id.store(update_id, Ordering::Relaxed);
                }

                let msg = &update["message"];
                let text = msg["text"].as_str().unwrap_or("");
                let chat_id = msg["chat"]["id"].as_i64().map(|id| id.to_string());
                let username = msg["from"]["username"].as_str().map(|s| s.to_string());
                let first_name = msg["from"]["first_name"].as_str().map(|s| s.to_string());

                if let Some(chat_id) = chat_id
                    && let Some(cmd) = text.strip_prefix('/')
                {
                    let cmd = cmd.split_whitespace().next().unwrap_or("");
                    // Strip @bot_name suffix (e.g. /stats@MyBot)
                    let cmd = cmd.split('@').next().unwrap_or(cmd);
                    if !cmd.is_empty() {
                        commands.push((
                            chat_id,
                            cmd.to_lowercase(),
                            username.clone(),
                            first_name.clone(),
                        ));
                    }
                }
            }
        }

        commands
    }
}
