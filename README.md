# Polymarket Signal Bot

> **Research project only.** This bot is built for learning and experimentation. It is not intended for production use, and you should not risk real money with it. Use at your own risk.

Automated prediction market trading bot that combines an XGBoost/ensemble ML model with Bayesian updating to find alpha on Polymarket. The model continuously retrains on its own resolved bets, improving over time.

## Architecture

```
src/
├── main.rs              # Entry point, multi-loop orchestration, graceful shutdown
├── bayesian.rs          # Bayesian updating with likelihood ratios
├── config.rs            # Env-var configuration (confique)
├── strategy.rs          # Strategy profiles (aggressive/balanced/conservative)
├── calibration.rs       # Calibration curve from resolved LLM estimates
├── format.rs            # Shared Telegram message formatting (open bets, truncation)
├── scanner/
│   ├── live.rs          # ML-first scanning, Bayesian consensus, signals
│   ├── ws.rs            # WebSocket price alerts for real-time triggers
│   └── news.rs          # Multi-source news fetching & keyword matching
├── data/
│   ├── models.rs        # GammaMarket, PriceTick data structures
│   └── crawler.rs       # Historical data crawler for backtesting
├── model/
│   ├── features.rs      # Feature engineering (momentum, volatility, RSI)
│   ├── xgb.rs           # Local XGBoost inference
│   └── sidecar.rs       # HTTP client for ensemble model server
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
    └── notifier.rs      # Telegram alerts, commands, broadcasts, GIF celebrations

scripts/
├── fetch_data.py        # Fetches resolved markets + own bets for training
├── train_model.py       # Trains XGBoost/stacking ensemble
├── serve_model.py       # HTTP model server (sidecar)
├── retrain.sh           # Continuous retraining loop (every 24h)
├── backtest.py          # Python backtest with stop-loss simulation
└── test_predictions.py  # Prediction validation tests
```

## How It Works

### ML-First Pipeline

1. **News scan loop** (every 10 min) fetches breaking news from Google News RSS, CBS Sports, CoinDesk, CoinTelegraph, Reuters/AP/BBC, and 15+ other feeds
2. Matches news to eligible markets by keyword relevance and embedding similarity (volume, expiry, price filters)
3. **XGBoost ensemble** scores all eligible markets on 15+ features (price momentum, volatility, volume trends, order book depth, time to expiry)
4. **Bayesian anchoring**: model predictions are anchored to market price as prior, with the model's likelihood ratio dampened by confidence (`LR^confidence`) — prevents overconfident predictions
5. Top candidates enriched with order book and price history data
6. Each **strategy profile** independently evaluates signals against its own thresholds and Kelly fraction
7. Places paper bets with per-strategy bankrolls and sends detailed Telegram summaries

### WebSocket Alerts

A parallel WebSocket connection monitors real-time price movements on eligible markets. Significant moves (3%+) trigger instant reassessment through the XGBoost model, enabling faster reaction than the 10-minute scan cycle. Open bet price alerts use a higher threshold (5%+ with 1h cooldown) to reduce noise.

### Continuous Learning

The model retrains every 24 hours on:
- **~1000 resolved Polymarket markets** fetched from the API
- **Own resolved bets** (weighted 3x) — both wins and losses, with exact entry prices and known outcomes

As more bets resolve, the model sees more of its own data and improves predictions over time.

### Risk Management

- **Stop-loss** (default disabled): exits positions when unrealized loss exceeds threshold
- **Expiry exit** (default disabled): exits underwater positions approaching expiry (guarded: 0 = off)
- **Terminal risk scaling**: reduces position size as market approaches expiry (`sqrt(days/14)`)
- **Per-strategy bankrolls**: each strategy has independent bankroll isolation
- **Graceful shutdown**: handles SIGTERM/SIGINT, sends Telegram notification before exit

### Housekeeping Loop

A separate loop (every 30 min) resolves settled bets, checks stop-loss/expiry exits, updates calibration data, logs predictions for Brier score tracking, and sends daily performance reports.

## Strategy Profiles

Three strategies run simultaneously, each with independent bankrolls (default €300 each):

| Strategy | Kelly | Min Edge | Min Confidence | Max Signals/Day |
|---|---|---|---|---|
| **Aggressive** | 50% | 5% | 40% | 10 |
| **Balanced** | 25% | 6% | 40% | 5 |
| **Conservative** | 15% | 8% | 50% | 3 |

XGBoost signals get relaxed thresholds (50% lower edge gate, 30% lower confidence gate) since the model has demonstrated calibration.

## Telegram

### Commands

| Command | Description |
|---|---|
| `/stats` | Portfolio statistics with per-strategy breakdown |
| `/open` | Open positions with live prices, PnL, Polymarket links |
| `/brier` | Model accuracy per source (Brier score vs market baseline) |
| `/health` | Bot uptime, scan counts, signals found |
| `/help` | List commands |

### Notifications

- **New bet**: strategy, side, stake, edge, confidence, reasoning
- **Bet resolved**: outcome, PnL, per-strategy record + random victory GIF on ~30% of wins
- **Price moves**: alerts on 5%+ moves on open bets (1h cooldown per market)
- **WS-triggered bets**: real-time bets from WebSocket price alerts
- **Heartbeat**: hourly summary with open bets, scan stats, strategy performance
- **Daily report**: full portfolio breakdown with Brier score

Subscribers receive all broadcasts. Shutdown notifications go to owner only.

## Setup

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Docker Compose (recommended)

```bash
docker compose up -d
```

Starts 4 services:
- **postgres**: data persistence
- **trainer**: continuous model retraining (every 24h)
- **model-server**: HTTP sidecar serving ensemble predictions
- **bot**: pre-built Rust binary from GHCR (`ghcr.io/skharchikov/polymarket-bot:latest`)

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
| `TELEGRAM_CHAT_ID` | *required* | Telegram chat ID (owner) |
| `OPENAI_API_KEY` | *required* | OpenAI API key (LLM fallback) |
| `MODEL_SIDECAR_URL` | `` | ML model server URL (auto in Docker) |
| `LLM_MODEL` | `gpt-4o` | LLM model for news assessment fallback |
| `STRATEGIES` | `aggressive,balanced,conservative` | Active strategy profiles |
| `STRATEGY_BANKROLL` | `300.0` | Starting bankroll per strategy (EUR) |
| `STRATEGY_MAX_SIGNALS` | `` | Per-strategy daily caps (e.g. `aggressive:10,balanced:5`) |
| `STOP_LOSS_PCT` | `999.0` | Stop-loss threshold (999.0 = disabled) |
| `EXIT_DAYS_BEFORE_EXPIRY` | `0` | Exit underwater positions N days before expiry (0 = disabled) |
| `CONSENSUS_AGENTS` | `2` | Number of LLM agents for fallback (1-3) |
| `SCAN_INTERVAL_MINS` | `30` | Housekeeping loop interval |
| `NEWS_SCAN_INTERVAL_MINS` | `10` | News scan loop interval |
| `MAX_MODEL_CANDIDATES` | `15` | Top N markets from XGBoost ranking |
| `MAX_LLM_CANDIDATES` | `1` | Markets assessed by LLM per cycle |
| `MIN_VOLUME` | `1000.0` | Min market volume filter |
| `MIN_BOOK_DEPTH` | `200.0` | Min order book depth (USD) |
| `MAX_DAYS_TO_EXPIRY` | `14` | Max days to market expiry |
| `KELLY_FRACTION` | `0.25` | Global Kelly fraction |
| `MIN_EFFECTIVE_EDGE` | `0.08` | Global min effective edge |
| `SLIPPAGE_PCT` | `0.01` | Slippage assumption (1%) |
| `FEE_PCT` | `0.02` | Trading fee (2%) |

## Prediction Tracking

Every prediction is logged to a `prediction_log` table with market price, model posterior, confidence, and edge. When markets resolve, Brier scores are computed per source (XGBoost vs LLM):

- **Model Brier** vs **Market Brier** shows whether the model adds value over just using market prices
- **Skill metric**: `1 - (model_brier / market_brier)` — positive means the model outperforms the market

## CI/CD

GitHub Actions with GHCR-based deployment:

- **CI** (on push/PR): `cargo fmt --check` → `cargo clippy` → `cargo test` in a single job
- **Docker** (on main push): builds musl release binary → pushes to `ghcr.io/skharchikov/polymarket-bot`
- **Release** (manual `workflow_dispatch`): bumps version, tags, creates GitHub release → triggers deploy
- **Deploy** (after release or manual): pulls image from GHCR, restarts services on Hetzner via SSH

The bot image is built once in CI (not on the server), cutting deploy time significantly.
