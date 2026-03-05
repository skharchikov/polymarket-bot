use anyhow::Result;
use rig::providers::openai;

pub async fn build_agent() -> Result<()> {
    // TODO: Configure with actual provider (OpenAI, Anthropic, etc.)
    // Example with OpenAI:
    //
    // let client = openai::Client::from_env();
    // let agent = client
    //     .agent("gpt-4o")
    //     .preamble("You are a prediction market analyst...")
    //     .tool(GetMarketsTool::new(MarketFetcher::new()))
    //     .build();

    tracing::info!("Agent module ready (configure provider to activate)");
    Ok(())
}
