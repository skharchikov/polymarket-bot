# Polymarket Signal Bot

Automated prediction market trading bot that finds alpha by matching breaking news to Polymarket markets that haven't priced it in yet.

Uses multi-agent LLM consensus (Skeptic + Catalyst + BaseRate roles) to assess probability impact, calibration tracking to correct systematic biases, and runs multiple strategy profiles simultaneously with independent bankrolls and Kelly Criterion sizing.

## Architecture

```
src/
├── main.rs              # Entry point, dual-loop orchestration
├── config.rs            # Env-var configuration (confique)
├── strategy.rs          # Strategy profiles (aggressive/balanced/conservative)
├── calibration.rs       # Calibration curve from resolved LLM estimates
├── scanner/
│   ├── live.rs          # Market scanning, multi-agent LLM consensus, signals
│   └── news.rs          # Multi-source news fetching & keyword matching
├── data/
│   ├── models.rs        # GammaMarket, PriceTick data structures
│   └── crawler.rs       # Historical data crawler for backtesting
├── pricing/
│   └── kelly.rs         # Kelly Criterion position sizing
├── storage/
│   ├── portfolio.rs     # Bet, PortfolioState, learning summary types
│   └── postgres.rs      # PostgreSQL persistence (per-strategy bankrolls)
├── backtest/
│   ├── engine.rs        # Backtest runner with pluggable strategies
│   ├── metrics.rs       # Sharpe ratio, Brier score, max drawdown
│   └── portfolio.rs     # In-memory portfolio for backtesting
└── telegram/
    └── notifier.rs      # Telegram alert formatting & delivery
```

## How It Works

1. **News scan loop** (every 10 min) fetches breaking news from Google News RSS, Reddit, and Polymarket trending
2. Matches news to eligible markets by keyword relevance (volume, expiry, price filters)
3. Checks order book liquidity and price history for top matches
4. **Multi-agent consensus**: 2-3 LLM agents (Skeptic, Catalyst, BaseRate) independently assess each market, aggregated via confidence-weighted averaging with disagreement penalty
5. **Calibration correction**: adjusts LLM estimates based on historical accuracy (10-bin curve with Laplace smoothing)
6. Each **strategy profile** independently evaluates signals against its own thresholds and Kelly fraction
7. Places paper bets with per-strategy bankrolls and sends Telegram alerts

A separate **housekeeping loop** (every 30 min) resolves settled bets, updates calibration data, calculates PnL, and sends daily performance reports.

## Strategy Profiles

Three strategies run simultaneously, each with independent bankrolls (default €300 each):

| Strategy | Kelly | Min Edge | Min Confidence | Max Signals/Day |
|---|---|---|---|---|
| **Aggressive** | 50% | 5% | 40% | 5 |
| **Balanced** | 25% | 8% | 50% | 3 |
| **Conservative** | 10% | 12% | 65% | 2 |

Strategies share the expensive LLM scan results but evaluate signals independently — no duplicate API calls.

## Setup

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Docker Compose (recommended)

```bash
docker compose up -d
```

Final image is ~3.5MB (scratch base, static musl binary, UPX compressed).

### Local Development

```bash
# Start Postgres
docker run -d \
  -e POSTGRES_DB=polymarket \
  -e POSTGRES_USER=bot \
  -e POSTGRES_PASSWORD=bot \
  -p 5432:5432 \
  postgres:17-alpine

# Run
cargo run                  # Live mode
cargo run -- test          # Test mode (2-min intervals)
cargo run -- backtest      # Backtest strategies on historical data
```

## Configuration

All settings via environment variables with sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | *required* | Postgres connection string |
| `TELEGRAM_BOT_TOKEN` | *required* | Telegram bot token |
| `TELEGRAM_CHAT_ID` | *required* | Telegram chat ID |
| `OPENAI_API_KEY` | *required* | OpenAI API key |
| `LLM_MODEL` | `gpt-4o` | OpenAI model for assessment |
| `STRATEGIES` | `aggressive,balanced,conservative` | Active strategy profiles |
| `STRATEGY_BANKROLL` | `300.0` | Starting bankroll per strategy (EUR) |
| `CONSENSUS_AGENTS` | `2` | Number of LLM agents (1-3) |
| `CONSENSUS_MAX_SPREAD` | `0.15` | Max agent disagreement before penalty |
| `CALIBRATION_MIN_SAMPLES` | `20` | Min resolved estimates before calibration activates |
| `SCAN_INTERVAL_MINS` | `30` | Housekeeping loop interval |
| `NEWS_SCAN_INTERVAL_MINS` | `10` | News scan loop interval |
| `HEARTBEAT_INTERVAL_MINS` | `60` | Heartbeat message interval (0 to disable) |
| `MAX_SIGNALS_PER_DAY` | `3` | Global max signals per day |
| `MAX_MARKETS_FETCH` | `1000` | Max markets to fetch from Polymarket API |
| `MAX_LLM_CANDIDATES` | `1` | Markets assessed per scan cycle |
| `MIN_VOLUME` | `1000.0` | Min market volume filter |
| `MIN_BOOK_DEPTH` | `200.0` | Min order book depth (USD) |
| `MIN_PRICE` | `0.03` | Min YES price filter |
| `MAX_PRICE` | `0.97` | Max YES price filter |
| `MAX_DAYS_TO_EXPIRY` | `30` | Max days to market expiry |
| `KELLY_FRACTION` | `0.25` | Global Kelly fraction (strategies override) |
| `MIN_EFFECTIVE_EDGE` | `0.08` | Global min edge (strategies override) |
| `SLIPPAGE_PCT` | `0.01` | Slippage assumption (1%) |
| `FEE_PCT` | `0.02` | Trading fee (2%) |
| `MIN_BET` | `10.0` | Minimum bet size (EUR) |

## Backtesting

Runs 4 strategies on historical closed markets:

- **SMA Crossover** — short vs long moving average momentum
- **Mean Reversion** — buy when price deviates >1 std-dev from mean
- **Trend Following** — linear regression slope with volatility filter
- **Contrarian Extremes** — counter-trend at prices far from 0.5

```bash
cargo run -- backtest
```

## Docker Image

Built with Alpine musl for a static binary, compressed with UPX, running on `scratch`:

- `strip = true`, `lto = true`, `codegen-units = 1`
- mimalloc allocator (musl's malloc is slow)
- Runs as unprivileged user (UID 10001)

## CI/CD

GitHub Actions on push to main:

- `cargo fmt --check`
- `cargo clippy --all-targets` (warnings = errors)
- `cargo build --release`
- `cargo test`
- Auto-deploy to Hetzner via SSH + Docker Compose
