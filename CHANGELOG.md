# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/skharchikov/polymarket-bot/releases/tag/v0.1.0) - 2026-03-08

### Other

- Fix formatting (cargo fmt)
- Remove unused CARGO_REGISTRY_TOKEN from release-plz workflow
- Add release-plz for automated releases and changelog
- Add Telegram scan cycle summary with rejection visibility
- Enable ANSI colors in logs, demote daily-limit log to debug
- Add CoinDesk/Reuters news sources, detailed bet decision logging
- Optimize rate limit waits: 25s->21s, remove redundant inter-candidate delay
- Switch from scratch to alpine for DNS resolution in Docker Compose
- Update README with strategies, consensus, calibration docs
- Retry DB connection up to 10 times on startup
- Remove redundant strategy field from AcceptedSignal
- Suppress dead_code warnings for CI -D warnings
- Optimize Docker image: Alpine musl build, scratch final, UPX + LTO
- Fix 5 bugs, add 16 tests, flat €300 bankroll per strategy
- Add multi-strategy profiles (aggressive, balanced, conservative)
- Relax market filters and make them configurable
- Add multi-agent consensus and calibration tracking
- Add configurable heartbeat messages and scan stats tracking
- Pin builder to rust:1.93-slim-bookworm to match runtime glibc
- Fix SSH key handling in deploy workflow
- Add CI/CD auto-deploy to Hetzner on push to main
- Add edge and confidence to bet resolution messages
- Improve Telegram messages with richer context
- Add README, rust-toolchain.toml, and clean up CI workflow
- Add GitHub CI pipeline and fix formatting/clippy issues
- Replace hardcoded constants with confique env-var configuration
- Split main loop into dual-loop: news scan (10min) + housekeeping (30min)
- Update rig-core to 0.32 and bump all dependencies
- Remove dead code, unused deps, and refactor for clean publish
- Feed past bet history into LLM for learning from mistakes
- Add test mode, Fear & Greed Index, and global market data for LLM
- Add crypto news feed, automatic bet resolution, and trending data
- Add CLI subcommands: cargo run for live bot, cargo run backtest for simulation
- Add portfolio_state.json to gitignore
- Add paper-trading portfolio with persistent state and daily reports
- Add live crypto prices, YES-only signals, and 7-day expiry filter
- Add live signal bot with Telegram notifications and LLM analysis
- Walk-forward backtest with observed-only history and 5 strategies
- Fix backtest for realistic simulation: first-tick entry, slippage, and fees
- Scale backtest to 2000 markets with paginated crawler and 4 strategies
- Rewrite strategies and wire up LLM probability estimator via Rig
- Fix crawler to use outcomePrices for resolution and add crypto filtering
- Fix Market deserialization to handle Polymarket string-encoded fields
- Add backtesting engine with portfolio simulation and metrics
- Add data layer: models and historical price crawler
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
