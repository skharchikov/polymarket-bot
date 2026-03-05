use anyhow::{Context, Result};
use reqwest::Client;

pub struct TelegramNotifier {
    client: Client,
    bot_token: String,
    chat_id: String,
}

impl TelegramNotifier {
    pub fn from_env() -> Result<Self> {
        let bot_token =
            std::env::var("TELEGRAM_BOT_TOKEN").context("TELEGRAM_BOT_TOKEN not set")?;
        let chat_id = std::env::var("TELEGRAM_CHAT_ID").context("TELEGRAM_CHAT_ID not set")?;

        Ok(Self {
            client: Client::new(),
            bot_token,
            chat_id,
        })
    }

    pub async fn send(&self, message: &str) -> Result<()> {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.bot_token);

        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": true,
            }))
            .send()
            .await
            .context("failed to send telegram message")?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Telegram API error: {body}");
        }

        tracing::info!("Telegram signal sent");
        Ok(())
    }
}
