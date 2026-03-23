# ADR 008: Cargo Workspace Split

**Status:** Accepted
**Date:** 2026-03-23

## Context

The project started as a single binary crate (`polymarket-bot`) that contained both the ML-driven trading pipeline and the copy-trading subsystem. Over time the two subsystems diverged in their runtime requirements:

- **ML trading bot** needs: rig-core (OpenAI/LLM), tokio-tungstenite (WebSocket), XGBoost model inference, sidecar HTTP client, backtest runner.
- **Copy trading bot** needs: none of the above — just the Polymarket data API, trader activity polling, and position mirroring logic.

Running both in one binary means every deployment carries the full dependency tree regardless of which features are used. It also makes independent scaling and separate deployment impossible.

## Decision

Split the repository into a Cargo workspace with three crates:

| Crate | Type | Purpose |
|-------|------|---------|
| `polymarket-common` | lib | Shared models, storage, pricing, Telegram notifier, metrics |
| `trading-bot` | bin | ML pipeline, LLM consensus, WebSocket alerts, backtest |
| `copy-trading-bot` | bin | Leaderboard monitoring, copy-trade execution |

### Shared library (`polymarket-common`)

Contains everything that both binaries need:
- `data::models` — `GammaMarket`, `PriceTick`, `fetch_yes_prices`, `fetch_market_by_slug`
- `data::crawler` — historical dataset builder
- `format` — Telegram message formatters
- `metrics` — Prometheus helpers
- `model::features` — `MarketFeatures` (needed by `storage::portfolio::NewBet`)
- `pricing::kelly` — Kelly criterion sizing
- `signal` — `SignalSource` enum (canonical, string-serialised)
- `storage::portfolio` — `NewBet`, `Bet`, `BetSide`, `CopyRef`
- `storage::postgres` — `PgPortfolio`, `RejectedSignal`, `NewCopyTradeEvent`
- `storage::copy_trade` — copy-trade storage helpers
- `telegram::notifier` — `TelegramNotifier`

### Copy-trading bot differences

The copy-trading bot's `cycles/copy_trade.rs` drops the informational `scanner.predict_market()` call present in the monolith. That call required `LiveScanner` (which pulls in rig-core and the full ML stack). The copy-trading bot has no ML model access and does not need it for its core function.

### Module re-export pattern

Both binaries use `pub use polymarket_common::<module>;` in `main.rs` so that internal code can use `crate::data`, `crate::storage`, etc. without path changes. Modules that are crate-local (e.g. `scanner`, `model`, `telegram::commands`) are declared with `mod` as usual.

### Migration path

`sqlx::migrate!` path in common changes from `"./migrations"` to `"../migrations"` since the common crate is one level below the workspace root.

`init_strategy_bankrolls` signature changed from `&[StrategyProfile]` to `&[String]` to remove the dependency on the trading-bot's `strategy` module from the shared library.

## Consequences

- Independent Docker images and deployments for each bot.
- Compile times improve because the copy-trading bot does not compile rig-core, tokio-tungstenite, or XGBoost dependencies.
- Two Telegram bot tokens are now required (one per binary).
- The `src/` directory is retained as the canonical source; new crate directories mirror its structure.
- `build.rs` (changelog/tag generation) lives in `trading-bot/` only; copy-trading-bot uses a default empty env var for `BUILD_TAG`/`BUILD_CHANGELOG`.
