use anyhow::Result;
use rig::completion::Chat;
use rig::providers::openai;

pub struct LlmEstimator {
    agent: rig::agent::Agent<openai::CompletionModel>,
}

impl LlmEstimator {
    pub fn new() -> Self {
        let client = openai::Client::from_env();
        let agent = client
            .agent("gpt-4o-mini")
            .preamble(
                "You are a prediction market probability estimator. \
                 Given a market question and its current price, estimate the TRUE probability \
                 that the outcome will be YES.\n\n\
                 Rules:\n\
                 - Respond with ONLY a JSON object: {\"probability\": 0.XX, \"confidence\": 0.XX}\n\
                 - probability: your estimate between 0.0 and 1.0\n\
                 - confidence: how confident you are in your estimate (0.0 = guessing, 1.0 = certain)\n\
                 - Be contrarian when you have evidence the market is wrong\n\
                 - Consider base rates, recent news, and logical reasoning\n\
                 - For crypto price markets, consider volatility and current trends\n\
                 - Do NOT just echo the market price back — that provides no value",
            )
            .temperature(0.2)
            .build();

        Self { agent }
    }

    pub async fn estimate(&self, question: &str, current_price: f64) -> Result<(f64, f64)> {
        let prompt = format!(
            "Market question: \"{question}\"\n\
             Current YES price: {current_price:.4}\n\
             Current implied probability: {:.1}%\n\n\
             What is the TRUE probability?",
            current_price * 100.0
        );

        let response = self.agent.chat(prompt, vec![]).await?;

        parse_estimate(&response)
    }
}

fn parse_estimate(response: &str) -> Result<(f64, f64)> {
    // Try to extract JSON from the response
    let json_str = if let Some(start) = response.find('{') {
        if let Some(end) = response[start..].find('}') {
            &response[start..=start + end]
        } else {
            response
        }
    } else {
        response
    };

    #[derive(serde::Deserialize)]
    struct Est {
        probability: f64,
        #[serde(default = "default_confidence")]
        confidence: f64,
    }

    fn default_confidence() -> f64 {
        0.5
    }

    let est: Est = serde_json::from_str(json_str)?;

    let prob = est.probability.clamp(0.01, 0.99);
    let conf = est.confidence.clamp(0.0, 1.0);

    Ok((prob, conf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_clean_json() {
        let (p, c) = parse_estimate(r#"{"probability": 0.72, "confidence": 0.8}"#).unwrap();
        assert!((p - 0.72).abs() < 1e-6);
        assert!((c - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_parse_json_in_text() {
        let resp = "Based on my analysis, here is my estimate:\n{\"probability\": 0.65, \"confidence\": 0.7}\nThat's my best guess.";
        let (p, _) = parse_estimate(resp).unwrap();
        assert!((p - 0.65).abs() < 1e-6);
    }

    #[test]
    fn test_clamps_extreme_values() {
        let (p, _) = parse_estimate(r#"{"probability": 1.5, "confidence": 2.0}"#).unwrap();
        assert!((p - 0.99).abs() < 1e-6);
    }
}
