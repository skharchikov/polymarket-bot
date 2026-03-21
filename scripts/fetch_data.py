#!/usr/bin/env python3
"""
Phase 1: Fetch resolved Polymarket data for model training.

Pulls historical markets from Gamma API and price history from CLOB API.
Stores raw data as JSON for the training pipeline.

Usage:
    python scripts/fetch_data.py [--markets 500] [--output model/training_data.json]
"""

import argparse
import re
import json
import os
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

# Sports markets are excluded from training — our live data shows consistent
# losses regardless of model version, and the model has no edge over sportsbooks.
_SPORTS_RE = re.compile(
    r"\b(nfl|nba|nhl|mlb|super\s+bowl|world\s+series|stanley\s+cup"
    r"|premier\s+league|champions\s+league|world\s+cup|la\s+liga|serie\s+a"
    r"|bundesliga|ligue\s+1|eredivisie|uefa|fifa"
    r"|ufc|mma|boxing|fight|bout|knockout"
    r"|formula\s*1|\bf1\b|nascar|motogp"
    r"|tennis|atp|wta|wimbledon|grand\s+slam"
    r"|basketball|soccer|football|hockey|baseball|cricket|rugby|golf"
    r"|win\s+on\s+202\d|\bvs\.\s|\bspread:|\bo/u\s+\d|moneyline"
    r"|match\s+result|correct\s+score)\b",
    re.IGNORECASE,
)


def is_sports(question: str, category: str) -> bool:
    return bool(_SPORTS_RE.search(f"{category} {question}"))


class PolymarketScraper:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PolymarketBot-Training/1.0",
            "Accept": "application/json",
        })
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._last_request = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request = time.time()

    def _get(self, url: str, params: dict = None) -> dict | list | None:
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if self.verbose:
                print(f"  [error] {url}: {e}", file=sys.stderr)
            return None

    def fetch_resolved_markets(self, limit: int = 500) -> list[dict]:
        """Fetch resolved (closed) markets from Gamma API."""
        markets = []
        offset = 0
        batch_size = 100

        while len(markets) < limit:
            if self.verbose:
                print(f"  Fetching markets offset={offset}...")
            data = self._get(f"{GAMMA_URL}/markets", params={
                "closed": "true",
                "limit": batch_size,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            })
            if not data or len(data) == 0:
                break

            for m in data:
                # Only keep binary markets with meaningful volume
                volume = float(m.get("volumeNum", 0) or m.get("volume", 0) or 0)
                if volume < 1000:
                    continue

                # Must have outcome prices to determine resolution
                prices_str = m.get("outcomePrices", "")
                if not prices_str:
                    continue

                try:
                    prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                    final_price = float(prices[0])
                except (json.JSONDecodeError, IndexError, TypeError):
                    continue

                # Determine resolved outcome: price >= 0.95 = YES, <= 0.05 = NO
                if final_price >= 0.95:
                    outcome = True
                elif final_price <= 0.05:
                    outcome = False
                else:
                    continue  # Skip unresolved or ambiguous

                # Skip sports markets — no model edge over sportsbooks
                question = m.get("question", "")
                category = m.get("groupSlug") or m.get("category", "") or ""
                if is_sports(question, category):
                    continue

                markets.append({
                    "market_id": m.get("id", ""),
                    "question": m.get("question", ""),
                    "category": m.get("groupSlug") or m.get("category", ""),
                    "outcome_yes": outcome,
                    "volume": volume,
                    # volume24hr reflects liquidity at a recent point, more
                    # representative of what live inference sees than total volume
                    "volume_24h": float(m.get("volume24hr", 0) or 0),
                    "liquidity": float(m.get("liquidityNum", 0) or m.get("liquidity", 0) or 0),
                    "end_date": m.get("endDate"),
                    "created_at": m.get("createdAt"),
                    "clob_token_ids": m.get("clobTokenIds", ""),
                    "slug": m.get("slug", ""),
                    "one_day_price_change": float(m.get("oneDayPriceChange", 0) or 0),
                    "one_week_price_change": float(m.get("oneWeekPriceChange", 0) or 0),
                })

            offset += batch_size
            if len(data) < batch_size:
                break

        return markets[:limit]

    def fetch_price_history(self, token_id: str, fidelity: int = 60) -> list[dict]:
        """Fetch price history ticks from CLOB API."""
        data = self._get(f"{CLOB_URL}/prices-history", params={
            "market": token_id,
            "interval": "max",
            "fidelity": fidelity,
        })
        if data and "history" in data:
            return data["history"]
        return []

    def fetch_orderbook(self, token_id: str) -> dict:
        """Fetch current orderbook for a token."""
        data = self._get(f"{CLOB_URL}/book", params={"token_id": token_id})
        if data:
            return {
                "bids": data.get("bids", []),
                "asks": data.get("asks", []),
            }
        return {"bids": [], "asks": []}

    def get_yes_token(self, market: dict) -> str | None:
        """Extract YES token ID from market."""
        tokens_raw = market.get("clob_token_ids", "")
        if not tokens_raw:
            return None
        try:
            tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
            return tokens[0] if tokens else None
        except (json.JSONDecodeError, IndexError):
            return None


def build_snapshots(scraper: PolymarketScraper, markets: list[dict],
                    verbose: bool = False) -> list[dict]:
    """
    For each market, fetch price history and generate feature snapshots
    at multiple time points.
    """
    snapshots = []

    for i, market in enumerate(markets):
        token_id = scraper.get_yes_token(market)
        if not token_id:
            continue

        if verbose:
            pct = (i + 1) / len(markets) * 100
            print(f"  [{i+1}/{len(markets)} {pct:.0f}%] {market['question'][:60]}...")

        history = scraper.fetch_price_history(token_id)
        if len(history) < 20:
            continue

        # Parse ticks
        ticks = []
        for tick in history:
            try:
                ticks.append({
                    "t": int(tick["t"]),
                    "p": float(tick["p"]),
                })
            except (KeyError, ValueError):
                continue

        if len(ticks) < 20:
            continue

        ticks.sort(key=lambda x: x["t"])
        prices = np.array([t["p"] for t in ticks])
        timestamps = np.array([t["t"] for t in ticks])

        # Generate snapshots every ~6h (or proportionally for shorter histories)
        total_duration = timestamps[-1] - timestamps[0]
        step = max(len(ticks) // 10, 5)  # At least 5 ticks between snapshots

        for idx in range(20, len(ticks) - 1, step):
            snap = _extract_snapshot(
                prices, timestamps, idx, market
            )
            if snap:
                snapshots.append(snap)

    return snapshots


def _extract_snapshot(prices: np.ndarray, timestamps: np.ndarray,
                      idx: int, market: dict) -> dict | None:
    """Extract feature vector at a point in time."""
    if idx < 10:
        return None

    current_price = prices[idx]

    # Skip extreme prices (already resolved)
    if current_price < 0.02 or current_price > 0.98:
        return None

    # Price features
    p_1h = _price_at_offset(prices, timestamps, idx, hours=1)
    p_6h = _price_at_offset(prices, timestamps, idx, hours=6)
    p_24h = _price_at_offset(prices, timestamps, idx, hours=24)

    momentum_1h = current_price - p_1h if p_1h else 0.0
    momentum_6h = current_price - p_6h if p_6h else 0.0
    momentum_24h = current_price - p_24h if p_24h else 0.0

    # Volatility: std of returns over last 24h of ticks
    lookback = min(idx, 24)
    window = prices[idx - lookback:idx + 1]
    if len(window) > 1:
        returns = np.diff(window) / (window[:-1] + 1e-10)
        volatility = float(np.std(returns))
    else:
        volatility = 0.0

    # Volatility 6h (for volatility_ratio)
    lookback_6h = min(idx, 6)
    window_6h = prices[idx - lookback_6h:idx + 1]
    if len(window_6h) > 1:
        returns_6h = np.diff(window_6h) / (window_6h[:-1] + 1e-10)
        volatility_6h = float(np.std(returns_6h))
    else:
        volatility_6h = 0.0
    volatility_ratio = min(volatility_6h / volatility, 5.0) if volatility > 1e-10 else 1.0

    # Trend consistency: fraction of last 14 ticks aligned with net direction
    period_tc = min(14, idx)
    if period_tc > 0:
        window_tc = prices[idx - period_tc:idx + 1]
        net_tc = window_tc[-1] - window_tc[0]
        if abs(net_tc) < 1e-10:
            trend_consistency = 0.5
        else:
            moves_tc = np.diff(window_tc)
            trend_consistency = float(np.sum(moves_tc * net_tc > 0) / len(moves_tc))
    else:
        trend_consistency = 0.5

    # RSI (14-period)
    rsi = _compute_rsi(prices[:idx + 1], period=14)

    # Volume features (from market metadata, static per market)
    volume = market.get("volume", 0)
    liquidity = market.get("liquidity", 0)

    # Days to expiry + temporal features
    end_date_str = market.get("end_date")
    created_at_str = market.get("created_at")
    snapshot_time = datetime.utcfromtimestamp(timestamps[idx])

    days_to_expiry = 30.0
    end_date = None
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).replace(tzinfo=None)
            days_to_expiry = max((end_date - snapshot_time).total_seconds() / 86400, 0)
        except (ValueError, TypeError):
            pass

    # Only keep snapshots within the deployment window: bot bets at 3-14d remaining.
    # Snapshots outside this range teach the model patterns that never appear in live.
    # Lower bound: sub-12h markets (5-min Bitcoin, hourly) are noise — never appear in live.
    if days_to_expiry > 14.0 or days_to_expiry < 0.5:
        return None

    created_date = None
    if created_at_str:
        try:
            created_date = datetime.fromisoformat(created_at_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            pass

    days_since_created = max((snapshot_time - created_date).total_seconds() / 86400, 0) \
        if created_date else 30.0
    created_to_expiry_span = max((end_date - created_date).total_seconds() / 86400, 0) \
        if (end_date and created_date) else 30.0

    # Category detection — must match live Rust logic (category + question, same keywords)
    category = (market.get("category") or "").lower()
    question = (market.get("question") or "").lower()
    combined = category + " " + question

    return {
        # Features (15 total — news/orderbook features removed: always 0 in training,
        # non-zero in live, causing distribution shift with no learning signal)
        "yes_price": current_price,
        "price_1h_ago": p_1h,
        "price_6h_ago": p_6h,
        "price_24h_ago": p_24h,
        "momentum_1h": momentum_1h,
        "momentum_6h": momentum_6h,
        "momentum_24h": momentum_24h,
        "volatility_24h": volatility,
        "volatility_ratio": volatility_ratio,
        "trend_consistency": trend_consistency,
        "rsi": rsi,
        "volume": market.get("volume_24h", volume),  # prefer 24h volume over total
        "liquidity": liquidity,
        "days_to_expiry": days_to_expiry,
        "days_since_created": days_since_created,
        "created_to_expiry_span": created_to_expiry_span,
        "category": combined,
        # Gamma API price changes
        "price_change_1d": market.get("one_day_price_change", momentum_24h),
        "price_change_1w": market.get("one_week_price_change", 0.0),
        # Label
        "outcome_yes": market["outcome_yes"],
        # Metadata
        "market_id": market["market_id"],
        "snapshot_ts": int(timestamps[idx]),
    }


def _price_at_offset(prices, timestamps, idx, hours):
    """Find price approximately `hours` before idx."""
    target_ts = timestamps[idx] - hours * 3600
    # Binary search for closest timestamp
    closest = np.searchsorted(timestamps[:idx], target_ts)
    if closest >= idx or closest < 0:
        return None
    return float(prices[closest])


def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute RSI (0-1 scale, not 0-100)."""
    if len(prices) < period + 1:
        return 0.5
    deltas = np.diff(prices[-period - 1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 1.0
    rs = avg_gain / avg_loss
    return 1.0 - (1.0 / (1.0 + rs))


def fetch_bet_snapshots(database_url: str, scraper: PolymarketScraper,
                        verbose: bool = False) -> list[dict]:
    """
    Pull resolved bets from our own DB and build snapshots from their
    price history at the time the bet was placed.

    These are the highest-quality training samples — we know the exact
    outcome AND had a specific thesis (edge, confidence) when entering.
    """
    try:
        import psycopg2
    except ImportError:
        print("  psycopg2 not installed, skipping DB bets (pip install psycopg2-binary)")
        return []

    try:
        conn = psycopg2.connect(database_url)
    except Exception as e:
        print(f"  Could not connect to DB: {e}")
        return []

    cur = conn.cursor()

    # First: use exact stored features from bet_features table (ADR 004).
    # These are the precise feature vector captured at bet-placement time.
    snapshots = []
    stored_market_ids = set()
    try:
        cur.execute("""
            SELECT b.market_id, b.side, b.entry_price, b.estimated_prob, b.confidence,
                   b.edge, b.won, bf.features, b.question
            FROM bets b
            JOIN bet_features bf ON bf.bet_id = b.id
            WHERE b.resolved = TRUE AND b.won IS NOT NULL
            ORDER BY b.placed_at
        """)
        for market_id, side, entry_price, est_prob, confidence, edge, won, features_json, question in cur.fetchall():
            if features_json is None:
                continue
            if is_sports(question or "", ""):
                continue
            outcome_yes = won if side == "Yes" else not won
            snap = {k: float(v) if v is not None else 0.0 for k, v in features_json.items()}
            snap["yes_price"] = entry_price
            snap["outcome_yes"] = outcome_yes
            snap["market_id"] = market_id
            snap["snapshot_ts"] = 0
            snap["source"] = "own_bet_exact"
            snap["our_estimated_prob"] = est_prob
            snap["our_confidence"] = confidence
            snap["our_edge"] = edge
            snap["our_bet_won"] = bool(won)
            snapshots.append(snap)
            stored_market_ids.add(market_id)
        if verbose and snapshots:
            print(f"  {len(snapshots)} bets with stored features (exact, no reconstruction)")
    except Exception as e:
        print(f"  Could not query bet_features: {e}")

    # Fallback: reconstruct from price history for older bets without stored features
    try:
        cur.execute("""
            SELECT market_id, side, entry_price, estimated_prob, confidence,
                   edge, won, placed_at, end_date, context
            FROM bets
            WHERE resolved = TRUE AND won IS NOT NULL
            ORDER BY placed_at
        """)
        rows = cur.fetchall()
    except Exception as e:
        print(f"  Could not query bets: {e}")
        conn.close()
        return snapshots
    conn.close()

    legacy_rows = [r for r in rows if r[0] not in stored_market_ids]
    if not legacy_rows:
        return snapshots

    print(f"  {len(legacy_rows)} older bets without stored features — reconstructing from price history")

    for i, row in enumerate(legacy_rows):
        market_id, side, entry_price, est_prob, confidence, edge, won, placed_at, end_date, ctx = row
        outcome_yes = won if side == "Yes" else not won

        # Skip sports bets from training data
        ctx_category = ctx.get("category", "") if ctx and isinstance(ctx, dict) else ""
        if is_sports("", ctx_category):
            continue

        token_id = None
        if ctx and isinstance(ctx, dict):
            token_id = ctx.get("yes_token_id")

        if not token_id:
            market_data = scraper._get(f"{GAMMA_URL}/markets/{market_id}")
            if market_data:
                tokens_raw = market_data.get("clobTokenIds", "")
                try:
                    tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
                    token_id = tokens[0] if tokens else None
                except (json.JSONDecodeError, IndexError):
                    pass

        if not token_id:
            continue

        if verbose:
            print(f"  [{i+1}/{len(legacy_rows)}] Reconstructing {market_id[:20]}... won={won}")

        history = scraper.fetch_price_history(token_id)
        if len(history) < 20:
            continue

        ticks = []
        for tick in history:
            try:
                ticks.append({"t": int(tick["t"]), "p": float(tick["p"])})
            except (KeyError, ValueError):
                continue

        if len(ticks) < 20:
            continue

        ticks.sort(key=lambda x: x["t"])
        prices = np.array([t["p"] for t in ticks])
        timestamps = np.array([t["t"] for t in ticks])

        bet_ts = int(placed_at.timestamp()) if hasattr(placed_at, "timestamp") else int(placed_at)
        bet_idx = int(np.searchsorted(timestamps, bet_ts))
        bet_idx = min(bet_idx, len(ticks) - 2)
        bet_idx = max(bet_idx, 20)

        volume = ctx.get("volume", 0) if ctx and isinstance(ctx, dict) else 0
        liquidity = ctx.get("liquidity", 0) if ctx and isinstance(ctx, dict) else 0

        market_stub = {
            "market_id": market_id,
            "question": "",
            "category": ctx.get("category", "") if ctx and isinstance(ctx, dict) else "",
            "outcome_yes": outcome_yes,
            "volume": volume,
            "liquidity": liquidity,
            "end_date": end_date,
        }

        snap = _extract_snapshot(prices, timestamps, bet_idx, market_stub)
        if snap:
            snap["yes_price"] = entry_price
            snap["source"] = "own_bet"
            snap["our_estimated_prob"] = est_prob
            snap["our_confidence"] = confidence
            snap["our_edge"] = edge
            snap["our_bet_won"] = bool(won)
            snapshots.append(snap)

    return snapshots


def main():
    parser = argparse.ArgumentParser(description="Fetch Polymarket training data")
    parser.add_argument("--markets", type=int, default=500,
                        help="Number of resolved markets to fetch")
    parser.add_argument("--output", type=str, default="model/training_data.json",
                        help="Output file path")
    parser.add_argument("--db", type=str, default=None,
                        help="DATABASE_URL to pull our own resolved bets")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Try DATABASE_URL from env if not passed
    database_url = args.db or os.environ.get("DATABASE_URL")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scraper = PolymarketScraper(verbose=args.verbose)

    # 1. Fetch resolved markets from Polymarket API
    print(f"Fetching {args.markets} resolved markets from Polymarket...")
    markets = scraper.fetch_resolved_markets(limit=args.markets)
    print(f"  Found {len(markets)} resolved markets with volume > $1k")

    print(f"Building feature snapshots from price history...")
    snapshots = build_snapshots(scraper, markets, verbose=args.verbose)
    print(f"  Generated {len(snapshots)} API snapshots")

    # 2. Pull our own resolved bets from DB
    if database_url:
        print(f"\nFetching our own resolved bets from DB...")
        bet_snapshots = fetch_bet_snapshots(database_url, scraper, verbose=args.verbose)
        if bet_snapshots:
            # Asymmetric weighting: losses 5x, wins 1x
            # Losses reveal model blind spots and deserve extra weight.
            # Wins included once — no upweighting to avoid confirmation bias.
            losses = [s for s in bet_snapshots if not s.get("our_bet_won", True)]
            wins = [s for s in bet_snapshots if s.get("our_bet_won", True)]
            print(f"  Adding {len(bet_snapshots)} bet snapshots "
                  f"({len(wins)} wins × 1, {len(losses)} losses × 5)")
            snapshots.extend(wins)
            for _ in range(5):
                snapshots.extend(losses)
    else:
        print("\nNo DATABASE_URL — skipping own bet data. "
              "Pass --db or set DATABASE_URL env var to include.")

    # Save
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(tz=None).isoformat(),
            "n_markets": len(markets),
            "n_snapshots": len(snapshots),
            "snapshots": snapshots,
        }, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Quick stats
    outcomes = [s["outcome_yes"] for s in snapshots]
    yes_pct = sum(outcomes) / len(outcomes) * 100 if outcomes else 0
    own = sum(1 for s in snapshots if s.get("source") == "own_bet")
    print(f"  Total: {len(snapshots)} snapshots ({own} from own bets)")
    print(f"  Class balance: {yes_pct:.1f}% YES / {100-yes_pct:.1f}% NO")


if __name__ == "__main__":
    main()
