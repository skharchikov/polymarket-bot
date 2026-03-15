#!/usr/bin/env python3
"""Quick test: load ensemble and predict on live Polymarket markets."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import requests

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
MODEL_DIR = Path("model")

FEATURE_NAMES = [
    "yes_price", "momentum_1h", "momentum_24h", "volatility_24h", "rsi",
    "log_volume", "log_liquidity", "days_to_expiry",
    "is_crypto", "is_politics", "is_sports",
    
    "price_change_1d", "price_change_1w",
]


def fetch_active_markets(n=100):
    all_markets = []
    offset = 0
    while len(all_markets) < n:
        resp = requests.get(f"{GAMMA_URL}/markets", params={
            "closed": "false", "active": "true", "limit": 100,
            "offset": offset,
            "order": "volumeNum", "ascending": "false",
        }, timeout=30)
        page = resp.json()
        if not page:
            break
        all_markets.extend(page)
        offset += len(page)
        if len(page) < 100:
            break
    return all_markets[:n]


def fetch_orderbook(token_id):
    try:
        resp = requests.get(f"{CLOB_URL}/book", params={"token_id": token_id}, timeout=10)
        data = resp.json()
        bid_vol, ask_vol, best_bid, best_ask = 0.0, 0.0, 0.0, 1.0
        for b in data.get("bids", []):
            p, s = float(b.get("price", 0)), float(b.get("size", 0))
            bid_vol += p * s
            best_bid = max(best_bid, p)
        for a in data.get("asks", []):
            p, s = float(a.get("price", 0)), float(a.get("size", 0))
            ask_vol += p * s
            best_ask = min(best_ask, p)
        total = bid_vol + ask_vol
        obi = (bid_vol - ask_vol) / total if total > 0 else 0.0
        spread = max(best_ask - best_bid, 0.0)
        return obi, spread, total
    except Exception:
        return 0.0, 0.0, 0.0


def fetch_price_history(token_id):
    try:
        resp = requests.get(f"{CLOB_URL}/prices-history",
                            params={"market": token_id, "interval": "max"}, timeout=15)
        data = resp.json()
        return data.get("history", [])
    except Exception:
        return []


def compute_features_from_history(history, current_price):
    """Compute momentum, volatility, RSI from real price history."""
    if len(history) < 2:
        return 0.0, 0.0, 0.0, 0.5  # mom1h, mom24h, vol, rsi

    ticks = [(int(t["t"]), float(t["p"])) for t in history]
    ticks.sort(key=lambda x: x[0])
    now_ts = ticks[-1][0]
    prices_arr = np.array([p for _, p in ticks])

    # Momentum
    def price_at_offset(offset_secs):
        target = now_ts - offset_secs
        idx = np.searchsorted([t for t, _ in ticks], target)
        if idx < len(ticks):
            return ticks[idx][1]
        return ticks[-1][1]

    p_1h = price_at_offset(3600)
    p_24h = price_at_offset(86400)
    mom_1h = current_price - p_1h
    mom_24h = current_price - p_24h

    # Volatility
    lookback = min(len(prices_arr), 24)
    window = prices_arr[-lookback:]
    if len(window) > 1:
        returns = np.diff(window) / (window[:-1] + 1e-10)
        vol = float(np.std(returns))
    else:
        vol = 0.0

    # RSI (14-period)
    period = 14
    if len(prices_arr) >= period + 1:
        deltas = np.diff(prices_arr[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain, avg_loss = np.mean(gains), np.mean(losses)
        if avg_loss == 0:
            rsi = 1.0
        else:
            rs = avg_gain / avg_loss
            rsi = 1.0 - (1.0 / (1.0 + rs))
    else:
        rsi = 0.5

    return mom_1h, mom_24h, vol, rsi


def build_features(m, obi, spread, history):
    from datetime import datetime, timezone
    prices_str = m.get("outcomePrices", "")
    try:
        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
        yes_price = float(prices[0])
    except (json.JSONDecodeError, IndexError, TypeError):
        return None

    if yes_price < 0.03 or yes_price > 0.97:
        return None

    volume = float(m.get("volumeNum", 0) or 0)
    if volume < 1000:
        return None
    liquidity = float(m.get("liquidityNum", 0) or 0)

    end_str = m.get("endDate")
    days_to_expiry = 30.0
    if end_str:
        try:
            end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            days_to_expiry = max((end - datetime.now(timezone.utc)).total_seconds() / 86400, 0)
        except (ValueError, TypeError):
            pass

    # Filter: only markets within 14 days (matching bot config)
    if days_to_expiry > 14:
        return None

    cat = (m.get("category") or "").lower()
    q = (m.get("question") or "").lower()
    crypto_kw = ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana"]
    is_crypto = 1.0 if any(k in q or k in cat for k in crypto_kw) else 0.0
    is_politics = 1.0 if any(k in q for k in ["election", "president", "vote"]) or "politic" in cat else 0.0
    is_sports = 1.0 if any(k in q for k in ["nba", "nfl", "soccer"]) or "sport" in cat else 0.0

    price_change_1d = float(m.get("oneDayPriceChange", 0) or 0)
    price_change_1w = float(m.get("oneWeekPriceChange", 0) or 0)

    mom_1h, mom_24h, vol, rsi = compute_features_from_history(history, yes_price)

    return [
        yes_price, mom_1h, mom_24h, vol, rsi,
        np.log(volume + 1), np.log(liquidity + 1), days_to_expiry,
        is_crypto, is_politics, is_sports,
        0.0, 0.0, 0.0,  # news features
        obi, spread, price_change_1d, price_change_1w,
    ], yes_price, days_to_expiry


def main():
    print("Loading ensemble model...")
    model = joblib.load(MODEL_DIR / "ensemble.joblib")
    scaler_path = MODEL_DIR / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    print(f"Model loaded: {type(model).__name__}\n")

    print("Fetching top 500 active markets...\n")
    markets = fetch_active_markets(500)

    results = []
    skipped_expiry = 0
    skipped_price = 0
    import time as _time
    for idx, m in enumerate(markets):
        tokens_raw = m.get("clobTokenIds", "")
        try:
            tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
            token_id = tokens[0] if tokens else None
        except (json.JSONDecodeError, IndexError):
            token_id = None

        if not token_id:
            continue

        # Quick pre-filter before expensive API calls
        prices_str = m.get("outcomePrices", "")
        try:
            prices_list = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
            quick_price = float(prices_list[0])
        except Exception:
            continue
        if quick_price < 0.03 or quick_price > 0.97:
            skipped_price += 1
            continue

        _time.sleep(0.1)  # rate limit
        obi, spread, depth = fetch_orderbook(token_id)
        _time.sleep(0.1)
        history = fetch_price_history(token_id)
        result = build_features(m, obi, spread, history)
        if result is None:
            skipped_expiry += 1
            continue
        fv, yes_price, days = result

        X = np.array(fv, dtype=np.float64).reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)

        prob = float(model.predict_proba(X)[0, 1])
        edge = prob - yes_price

        # Estimate confidence from base model disagreement
        confidence = 0.50
        try:
            estimator = model
            if hasattr(estimator, "calibrated_classifiers_"):
                estimator = estimator.calibrated_classifiers_[0].estimator
            if hasattr(estimator, "estimators_"):
                base_preds = []
                for est in estimator.estimators_:
                    if isinstance(est, list):
                        for e in est:
                            base_preds.append(float(e.predict_proba(X)[0, 1]))
                    else:
                        base_preds.append(float(est.predict_proba(X)[0, 1]))
                if len(base_preds) >= 2:
                    sp = max(base_preds) - min(base_preds)
                    confidence = max(0.25, min(0.75, 0.75 / (1.0 + sp * 4.0)))
        except Exception:
            pass

        results.append({
            "question": m.get("question", "")[:70],
            "price": yes_price,
            "prob": prob,
            "edge": edge,
            "conf": confidence,
            "eff_edge": abs(edge) * confidence,
            "obi": obi,
            "spread": spread,
            "depth": depth,
            "volume": float(m.get("volumeNum", 0) or 0),
            "days": days,
        })

        if len(results) % 5 == 0:
            print(f"  Processed {len(results)} eligible markets ({idx+1}/{len(markets)} scanned)...")

    # Sort by effective edge descending
    results.sort(key=lambda r: r["eff_edge"], reverse=True)

    print(f"\nSkipped: {skipped_price} (extreme price), {skipped_expiry} (>14d expiry or low volume)")
    print(f"\n{'Market':<60} {'Price':>6} {'Prob':>6} {'Edge':>7} {'Conf':>5} {'EffEdge':>7} {'Days':>5} {'OBI':>6} {'Sprd':>5} {'Depth':>8}")
    print("-" * 135)
    for r in results:
        side = "YES" if r["edge"] > 0 else "NO "
        print(
            f"{r['question']:<60} {r['price']:5.1%} {r['prob']:5.1%} "
            f"{side} {abs(r['edge']):+5.1%} {r['conf']:4.0%} {r['eff_edge']:5.1%} "
            f"{r['days']:5.1f} {r['obi']:+5.2f} {r['spread']:4.3f} ${r['depth']:>7,.0f}"
        )

    # Summary
    actionable = [r for r in results if r["eff_edge"] >= 0.05]
    print(f"\n{len(actionable)}/{len(results)} markets with effective edge >= 5%")
    if actionable:
        print("\nTop signals:")
        for r in actionable[:5]:
            side = "YES" if r["edge"] > 0 else "NO"
            print(f"  {side} {r['question'][:60]}  edge={abs(r['edge']):+.1%} conf={r['conf']:.0%}")


if __name__ == "__main__":
    main()
