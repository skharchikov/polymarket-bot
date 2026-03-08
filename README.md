# Polymarket Signal Bot

Automated prediction market trading bot that finds alpha by identifying breaking news that Polymarket hasn't priced in yet.

Scans news from multiple sources, matches them to active markets, uses GPT-4o to assess probability impact, and places paper bets with Kelly Criterion sizing.

## Architecture

```
src/
├── main.rs              # Entry point, dual-loop orchestration
├── config.rs            # Env-var configuration (confique)
├── scanner/
│   ├── live.rs          # Market scanning, LLM assessment, signal generation
│   └── news.rs          # Multi-source news fetching & keyword matching
├── data/
│   ├── models.rs        # GammaMarket, PriceTick data structures
│   └── crawler.rs       # Historical data crawler for backtesting
├── pricing/
│   └── kelly.rs         # Kelly Criterion position sizing
├── storage/
│   ├── portfolio.rs     # Bet, PortfolioState, DailySnapshot types
│   └── postgres.rs      # PostgreSQL persistence layer
├── backtest/
│   ├── engine.rs        # Backtest runner with pluggable strategies
│   ├── metrics.rs       # Sharpe ratio, Brier score, max drawdown
│   └── portfolio.rs     # In-memory portfolio for backtesting
└── telegram/
    └── notifier.rs      # Telegram alert formatting & delivery
```

## How It Works

1. **News scan loop** (every 10 min) fetches breaking news from Google News RSS, Reddit, and Polymarket trending
2. Matches news to active markets by keyword relevance (≥2 keyword overlap)
3. Checks order book liquidity and price history for top matches
4. GPT-4o assesses whether news shifts the true probability vs current market price
5. Calculates edge and applies fractional Kelly Criterion for position sizing
6. Places paper bets and sends Telegram alerts

A separate **housekeeping loop** (every 30 min) resolves settled bets, calculates PnL, and sends daily performance reports.

Headlines are deduplicated across scan cycles to avoid wasting LLM calls on stale news.

## Setup

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Docker Compose (recommended)

```bash
docker-compose up -d
```

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
| `OPENAI_API_KEY` | *required* | OpenAI API key (set externally) |
| `SCAN_INTERVAL_MINS` | `30` | Housekeeping loop interval |
| `NEWS_SCAN_INTERVAL_MINS` | `10` | News scan loop interval |
| `MAX_SIGNALS_PER_DAY` | `3` | Max bets per day |
| `SLIPPAGE_PCT` | `0.01` | Slippage assumption (1%) |
| `FEE_PCT` | `0.02` | Trading fee (2%) |
| `MIN_BET` | `10.0` | Minimum bet size (EUR) |
| `MIN_VOLUME` | `5000.0` | Min market volume filter |
| `MIN_BOOK_DEPTH` | `200.0` | Min order book depth (USD) |
| `KELLY_FRACTION` | `0.25` | Quarter-Kelly for safety |
| `MAX_DAYS_TO_EXPIRY` | `14` | Max days to market expiry |
| `MAX_LLM_CANDIDATES` | `3` | Markets assessed per scan |
| `MIN_EFFECTIVE_EDGE` | `0.08` | Min edge*confidence threshold |
| `LLM_MODEL` | `gpt-4o` | OpenAI model for assessment |

## Backtesting

Runs 4 strategies on historical closed markets:

- **SMA Crossover** — short vs long moving average momentum
- **Mean Reversion** — buy when price deviates >1 std-dev from mean
- **Trend Following** — linear regression slope with volatility filter
- **Contrarian Extremes** — counter-trend at prices far from 0.5

Reports ROI, win rate, Sharpe ratio, Brier score, and max drawdown for each.

```bash
cargo run -- backtest
```

## CI

GitHub Actions runs on push to main/master and on PRs:

- `cargo fmt --check`
- `cargo clippy --all-targets` (warnings = errors)
- `cargo build --release`
- `cargo test`
