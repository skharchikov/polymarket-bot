use alloy::signers::local::PrivateKeySigner;
use anyhow::Result;

pub struct Wallet {
    pub signer: PrivateKeySigner,
}

impl Wallet {
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("PRIVATE_KEY")?;
        let signer: PrivateKeySigner = key.parse()?;
        tracing::info!(address = %signer.address(), "Wallet loaded");
        Ok(Self { signer })
    }
}
