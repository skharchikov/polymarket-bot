#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalSource {
    XgBoost,
    LlmConsensus,
    CopyTrade,
}

impl SignalSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::XgBoost => "xgboost",
            Self::LlmConsensus => "llm_consensus",
            Self::CopyTrade => "copy_trade",
        }
    }
}
