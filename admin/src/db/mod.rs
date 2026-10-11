//! Read-only queries for admin pages (admin_ro role).

/// SQL expression classifying a bet row into its bot.
pub const BOT_EXPR: &str = "CASE WHEN strategy LIKE 'copy:%' THEN 'copy' ELSE 'trading' END";

/// Phase 1 only shows paper money; the mode switch arrives with live trading.
pub const DEFAULT_MODE: &str = "paper";
pub mod history;
pub mod overview;
pub mod positions;
pub mod status;
