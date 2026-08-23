use anyhow::{Context, Result};
use reqwest::Client;
use std::sync::atomic::{AtomicI64, Ordering};

/// Outcome of a failed Telegram send, used to decide whether to prune a subscriber.
#[derive(Debug)]
pub enum SendError {
    /// The chat is gone for this bot (blocked / not found / deactivated) — prune it.
    Permanent(String),
    /// Temporary failure (rate limit, server error, network) — keep the subscriber.
    Transient(String),
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendError::Permanent(m) => write!(f, "permanent: {m}"),
            SendError::Transient(m) => write!(f, "transient: {m}"),
        }
    }
}

impl std::error::Error for SendError {}

/// Classify a Telegram sendMessage failure. `error_code` is Telegram's numeric
/// code (None for a transport/network error). 400 (chat not found) and 403
/// (bot blocked / user deactivated) mean the chat is permanently unreachable.
pub fn classify_telegram_error(error_code: Option<i64>, body: &str) -> SendError {
    match error_code {
        Some(400) | Some(403) => SendError::Permanent(body.to_string()),
        _ => SendError::Transient(body.to_string()),
    }
}

pub struct TelegramNotifier {
    client: Client,
    bot_token: String,
    chat_id: String,
    bot_kind: String,
    /// Last processed update_id for polling
    last_update_id: AtomicI64,
}

impl TelegramNotifier {
    pub fn new(bot_token: &str, chat_id: &str, bot_kind: &str) -> Self {
        Self {
            client: Client::new(),
            bot_token: bot_token.to_string(),
            chat_id: chat_id.to_string(),
            bot_kind: bot_kind.to_string(),
            last_update_id: AtomicI64::new(0),
        }
    }

    /// The bot identity ("trading" or "copy") used to scope subscribers.
    pub fn bot_kind(&self) -> &str {
        &self.bot_kind
    }

    /// Check if a chat_id belongs to the bot owner.
    pub fn is_owner(&self, chat_id: &str) -> bool {
        chat_id == self.chat_id
    }

    pub async fn send(&self, message: &str) -> Result<()> {
        self.send_to(&self.chat_id, message).await
    }

    /// Send to owner + all subscribers. Returns chat_ids that PERMANENTLY failed
    /// (blocked / chat not found) so the caller can deactivate them.
    pub async fn broadcast(
        &self,
        subscribers: &[(String, Option<String>)],
        message: &str,
    ) -> Vec<String> {
        let _ = self.send(message).await; // owner
        let mut pruned = Vec::new();
        for (id, username) in subscribers {
            if id == &self.chat_id {
                continue;
            }
            let label = username.as_deref().unwrap_or("unknown");
            match self.send_to_classified(id, message).await {
                Ok(()) => {}
                Err(SendError::Permanent(body)) => {
                    tracing::info!(chat_id = %id, username = label, "Pruning unreachable subscriber");
                    tracing::debug!(chat_id = %id, body = %body, "Permanent Telegram failure");
                    pruned.push(id.clone());
                }
                Err(SendError::Transient(body)) => {
                    tracing::warn!(chat_id = %id, username = label, err = %body, "Transient send failure (kept)");
                }
            }
        }
        pruned
    }

    /// Send to one chat, returning a typed error that says whether the failure
    /// is permanent (prune the chat) or transient (keep it).
    async fn send_to_classified(
        &self,
        chat_id: &str,
        message: &str,
    ) -> std::result::Result<(), SendError> {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.bot_token);
        let disclaimer = "_This is not financial advice. Do your own research._";
        let footer = format!("\n\n{disclaimer}");
        let chunks = split_message(message, 4096 - footer.len());

        for (i, chunk) in chunks.iter().enumerate() {
            let text = if i == chunks.len() - 1 {
                format!("{chunk}{footer}")
            } else {
                chunk.to_string()
            };
            let resp = match self
                .client
                .post(&url)
                .json(&serde_json::json!({
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": true,
                }))
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => return Err(SendError::Transient(e.to_string())),
            };
            if !resp.status().is_success() {
                let body = resp.text().await.unwrap_or_default();
                let code = serde_json::from_str::<serde_json::Value>(&body)
                    .ok()
                    .and_then(|v| v["error_code"].as_i64());
                return Err(classify_telegram_error(code, &body));
            }
        }
        Ok(())
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
        let footer = format!("\n\n{disclaimer}");
        let chunks = split_message(message, 4096 - footer.len());

        for (i, chunk) in chunks.iter().enumerate() {
            let text = if i == chunks.len() - 1 {
                format!("{chunk}{footer}")
            } else {
                chunk.to_string()
            };

            let resp = self
                .client
                .post(&url)
                .json(&serde_json::json!({
                    "chat_id": chat_id,
                    "text": text,
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

    /// Poll for new commands. Returns (chat_id, command, username, first_name, full_text).
    pub async fn poll_commands(
        &self,
    ) -> Vec<(String, String, Option<String>, Option<String>, String)> {
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
                            text.to_string(),
                        ));
                    }
                }
            }
        }

        commands
    }
}

/// Split `text` into chunks of at most `max_chars` characters, breaking at
/// newline boundaries where possible. Each chunk is guaranteed ≤ `max_chars`.
fn split_message(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in text.split('\n') {
        let line_with_newline = if current.is_empty() {
            line.to_string()
        } else {
            format!("\n{line}")
        };

        if current.chars().count() + line_with_newline.chars().count() > max_chars {
            if !current.is_empty() {
                chunks.push(current.clone());
                current.clear();
            }
            // Line itself may exceed limit — hard-split by chars
            let mut remaining = line;
            while !remaining.is_empty() {
                let take: String = remaining.chars().take(max_chars).collect();
                let byte_len = take.len();
                chunks.push(take);
                remaining = &remaining[byte_len..];
            }
        } else {
            current.push_str(&line_with_newline);
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_blocked_is_permanent() {
        let body = r#"{"ok":false,"error_code":403,"description":"Forbidden: bot was blocked by the user"}"#;
        assert!(matches!(
            classify_telegram_error(Some(403), body),
            SendError::Permanent(_)
        ));
    }

    #[test]
    fn test_classify_chat_not_found_is_permanent() {
        let body = r#"{"ok":false,"error_code":400,"description":"Bad Request: chat not found"}"#;
        assert!(matches!(
            classify_telegram_error(Some(400), body),
            SendError::Permanent(_)
        ));
    }

    #[test]
    fn test_classify_rate_limit_is_transient() {
        let body = r#"{"ok":false,"error_code":429,"description":"Too Many Requests"}"#;
        assert!(matches!(
            classify_telegram_error(Some(429), body),
            SendError::Transient(_)
        ));
    }

    #[test]
    fn test_classify_unknown_is_transient() {
        assert!(matches!(
            classify_telegram_error(None, "network error"),
            SendError::Transient(_)
        ));
    }

    #[test]
    fn test_split_short_message_unchanged() {
        let msg = "Hello world";
        let chunks = split_message(msg, 100);
        assert_eq!(chunks, vec!["Hello world"]);
    }

    #[test]
    fn test_split_at_newline_boundary() {
        let msg = "line one\nline two\nline three";
        // limit forces split after "line one"
        let chunks = split_message(msg, 10);
        assert!(chunks.iter().all(|c| c.chars().count() <= 10));
        // recombined content equals original
        assert_eq!(chunks.join("\n"), msg);
    }

    #[test]
    fn test_split_preserves_all_content() {
        let msg = (0..100)
            .map(|i| format!("Line {i}: some content here"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = split_message(&msg, 200);
        assert!(chunks.iter().all(|c| c.chars().count() <= 200));
        assert_eq!(chunks.join("\n"), msg);
    }

    #[test]
    fn test_split_exact_limit_no_split() {
        let msg = "abcde";
        let chunks = split_message(msg, 5);
        assert_eq!(chunks, vec!["abcde"]);
    }
}
