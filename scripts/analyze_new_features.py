#!/usr/bin/env python3
"""
Analyze effect of v4 features (momentum_6h, volatility_ratio, trend_consistency)
on our resolved bets by reconstructing them from price history at bet-placement time.

Usage:
    python scripts/analyze_new_features.py --db postgresql://...
    # or set DATABASE_URL env var
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CLOB_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"


def make_session():
    s = requests.Session()
    s.headers["User-Agent"] = "PolymarketBot-Analysis/1.0"
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


_last = 0


def rate_limited_get(session, url, params=None):
    global _last
    elapsed = time.time() - _last
    if elapsed < 0.5:
        time.sleep(0.5 - elapsed)
    _last = time.time()
    try:
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [warn] {url}: {e}", file=sys.stderr)
        return None


def fetch_price_history(session, token_id):
    data = rate_limited_get(session, f"{CLOB_URL}/prices-history", {
        "market": token_id, "interval": "max", "fidelity": 60,
    })
    if data and "history" in data:
        ticks = []
        for t in data["history"]:
            try:
                ticks.append({"t": int(t["t"]), "p": float(t["p"])})
            except (KeyError, ValueError):
                continue
        ticks.sort(key=lambda x: x["t"])
        return ticks
    return []


def price_at_offset(prices, timestamps, idx, hours):
    target = timestamps[idx] - hours * 3600
    closest = np.searchsorted(timestamps[:idx], target)
    if closest >= idx or closest < 0:
        return None
    return float(prices[closest])


def compute_volatility(prices, idx, hours):
    lookback = min(idx, hours)
    window = prices[idx - lookback:idx + 1]
    if len(window) < 2:
        return 0.0
    returns = np.diff(window) / (window[:-1] + 1e-10)
    return float(np.std(returns))


def compute_trend_consistency(prices, idx, period=14):
    p = min(period, idx)
    if p == 0:
        return 0.5
    window = prices[idx - p:idx + 1]
    net = window[-1] - window[0]
    if abs(net) < 1e-10:
        return 0.5
    moves = np.diff(window)
    return float(np.sum(moves * net > 0) / len(moves))


def compute_new_features(ticks, bet_ts):
    if len(ticks) < 10:
        return None

    prices = np.array([t["p"] for t in ticks])
    timestamps = np.array([t["t"] for t in ticks])

    idx = int(np.searchsorted(timestamps, bet_ts))
    idx = min(idx, len(ticks) - 2)
    idx = max(idx, 10)

    current = prices[idx]
    p_6h = price_at_offset(prices, timestamps, idx, hours=6)
    momentum_6h = current - p_6h if p_6h is not None else 0.0

    vol_24h = compute_volatility(prices, idx, hours=24)
    vol_6h = compute_volatility(prices, idx, hours=6)
    volatility_ratio = min(vol_6h / vol_24h, 5.0) if vol_24h > 1e-10 else 1.0

    trend_consistency = compute_trend_consistency(prices, idx, period=14)

    return {
        "momentum_6h": momentum_6h,
        "volatility_ratio": volatility_ratio,
        "trend_consistency": trend_consistency,
        "entry_price_reconstructed": current,
    }


def load_bets(database_url):
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("pip install psycopg2-binary")
        sys.exit(1)

    conn = psycopg2.connect(database_url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT b.id, b.market_id, b.question, b.side, b.entry_price,
               b.edge, b.confidence, b.won, b.placed_at, b.end_date,
               b.strategy, b.context, bf.features
        FROM bets b
        LEFT JOIN bet_features bf ON bf.bet_id = b.id
        WHERE b.resolved = TRUE AND b.won IS NOT NULL
        ORDER BY b.placed_at
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def get_token_id(session, row):
    ctx = row.get("context")
    if ctx and isinstance(ctx, dict):
        tid = ctx.get("yes_token_id")
        if tid:
            return tid

    market_id = row["market_id"]
    data = rate_limited_get(session, f"{GAMMA_URL}/markets/{market_id}")
    if data:
        raw = data.get("clobTokenIds", "")
        try:
            tokens = json.loads(raw) if isinstance(raw, str) else raw
            return tokens[0] if tokens else None
        except (json.JSONDecodeError, IndexError):
            pass
    return None


def print_bucket_analysis(label, values_win, values_lose, unit=""):
    if not values_win and not values_lose:
        return
    all_vals = values_win + values_lose
    lo, hi = min(all_vals), max(all_vals)
    if abs(hi - lo) < 1e-10:
        print(f"  {label}: no variance")
        return

    n_buckets = 4
    edges = np.linspace(lo, hi, n_buckets + 1)

    print(f"\n  {label}{unit}:")
    print(f"  {'Bucket':<22} {'Wins':>5} {'Losses':>6} {'Win%':>6} {'Avg edge':>9}")
    for i in range(n_buckets):
        lo_b, hi_b = edges[i], edges[i + 1]
        w = sum(1 for v in values_win if lo_b <= v <= hi_b)
        l = sum(1 for v in values_lose if lo_b <= v <= hi_b)
        total = w + l
        if total == 0:
            continue
        win_pct = w / total * 100
        label_b = f"[{lo_b:+.3f}, {hi_b:+.3f}]"
        print(f"  {label_b:<22} {w:>5} {l:>6} {win_pct:>5.0f}% {'':>9}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()

    if not args.db:
        print("--db or DATABASE_URL required")
        sys.exit(1)

    print("Loading resolved bets from DB...")
    rows = load_bets(args.db)
    print(f"  {len(rows)} resolved bets")

    session = make_session()

    results = []
    for i, row in enumerate(rows):
        print(f"  [{i+1}/{len(rows)}] {row['question'][:60]}... won={row['won']}", end="\r")

        token_id = get_token_id(session, row)
        if not token_id:
            continue

        placed_at = row["placed_at"]
        bet_ts = int(placed_at.timestamp()) if hasattr(placed_at, "timestamp") else int(placed_at)

        ticks = fetch_price_history(session, token_id)
        if len(ticks) < 10:
            continue

        feats = compute_new_features(ticks, bet_ts)
        if feats is None:
            continue

        side = row["side"]
        won = row["won"]
        outcome_yes = won if side == "Yes" else not won

        # Also pull stored features if available
        stored = row.get("features") or {}

        results.append({
            "question": row["question"],
            "side": side,
            "won": won,
            "outcome_yes": outcome_yes,
            "entry_price": row["entry_price"],
            "edge": row["edge"],
            "confidence": row["confidence"],
            "strategy": row["strategy"],
            **feats,
            # Stored features (may be missing new ones)
            "rsi": stored.get("rsi", 0.0),
            "yes_price_stored": stored.get("yes_price", row["entry_price"]),
            "volatility_24h": stored.get("volatility_24h", 0.0),
            "momentum_1h_stored": stored.get("momentum_1h", 0.0),
            "momentum_24h_stored": stored.get("momentum_24h", 0.0),
        })

    print(f"\nReconstructed features for {len(results)} bets")

    wins = [r for r in results if r["won"]]
    losses = [r for r in results if not r["won"]]
    print(f"  Wins: {len(wins)}  Losses: {len(losses)}  Win rate: {len(wins)/len(results)*100:.1f}%\n")

    # --- Analysis ---
    print("=" * 60)
    print("NEW FEATURE DISTRIBUTIONS: WINS vs LOSSES")
    print("=" * 60)

    for feat, unit in [
        ("momentum_6h", ""),
        ("volatility_ratio", ""),
        ("trend_consistency", ""),
    ]:
        w_vals = [r[feat] for r in wins]
        l_vals = [r[feat] for r in losses]
        w_mean = np.mean(w_vals) if w_vals else 0
        l_mean = np.mean(l_vals) if l_vals else 0
        print(f"\n{feat}:")
        print(f"  Wins   mean={w_mean:+.4f}  std={np.std(w_vals):.4f}  median={np.median(w_vals):+.4f}")
        print(f"  Losses mean={l_mean:+.4f}  std={np.std(l_vals):.4f}  median={np.median(l_vals):+.4f}")
        print_bucket_analysis(feat, w_vals, l_vals, unit)

    # Check: momentum_6h aligned with bet direction?
    print("\n" + "=" * 60)
    print("MOMENTUM_6H ALIGNMENT WITH BET SIDE")
    print("=" * 60)
    aligned_wins, aligned_losses = 0, 0
    opposed_wins, opposed_losses = 0, 0
    for r in results:
        m6 = r["momentum_6h"]
        # Aligned = momentum in direction of bet (YES bet + positive momentum, NO bet + negative momentum)
        aligned = (r["side"] == "Yes" and m6 > 0) or (r["side"] == "No" and m6 < 0)
        if aligned:
            if r["won"]: aligned_wins += 1
            else: aligned_losses += 1
        else:
            if r["won"]: opposed_wins += 1
            else: opposed_losses += 1

    def pct(w, l): return f"{w/(w+l)*100:.0f}%" if w+l else "n/a"
    print(f"  Momentum aligned with bet: {aligned_wins}W / {aligned_losses}L = {pct(aligned_wins, aligned_losses)} win rate")
    print(f"  Momentum opposed to bet:   {opposed_wins}W / {opposed_losses}L = {pct(opposed_wins, opposed_losses)} win rate")

    # Volatility regime analysis
    print("\n" + "=" * 60)
    print("VOLATILITY REGIME (volatility_ratio vs 1.0)")
    print("=" * 60)
    expanding = [r for r in results if r["volatility_ratio"] > 1.0]
    settling  = [r for r in results if r["volatility_ratio"] <= 1.0]
    exp_w = sum(1 for r in expanding if r["won"])
    set_w = sum(1 for r in settling  if r["won"])
    print(f"  Expanding vol (ratio>1): {exp_w}W / {len(expanding)-exp_w}L = {pct(exp_w, len(expanding)-exp_w)} win rate  (n={len(expanding)})")
    print(f"  Settling vol  (ratio≤1): {set_w}W / {len(settling)-set_w}L  = {pct(set_w, len(settling)-set_w)} win rate  (n={len(settling)})")

    # Trend consistency buckets
    print("\n" + "=" * 60)
    print("TREND CONSISTENCY BUCKETS")
    print("=" * 60)
    for lo, hi, label in [(0.0, 0.4, "Choppy <0.4"), (0.4, 0.6, "Neutral 0.4-0.6"), (0.6, 1.01, "Trending >0.6")]:
        bucket = [r for r in results if lo <= r["trend_consistency"] < hi]
        bw = sum(1 for r in bucket if r["won"])
        print(f"  {label:<20}: {bw}W / {len(bucket)-bw}L = {pct(bw, len(bucket)-bw)} win rate  (n={len(bucket)})")

    print()


if __name__ == "__main__":
    main()
