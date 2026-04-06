#!/usr/bin/env python3
"""
Build training data from the Jon-Becker prediction-market-analysis dataset.

Uses DuckDB to efficiently join 310M+ trades with resolved markets,
sample representative snapshots, and extract features.

Outputs training_data_v5.json compatible with train_model.py.

Usage:
    python scripts/train_from_dataset.py \
        --data-dir /mnt/HC_Volume_105339755/data/polymarket \
        --output /mnt/HC_Volume_105339755/training_data_v5.json \
        --max-samples 50000 \
        -v
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("pip install duckdb", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# NLP feature extraction
# ---------------------------------------------------------------------------

SPORTS_RE = re.compile(r" vs\. | vs |spread:|o/u |over/under|win on 2", re.IGNORECASE)
CRYPTO_RE = re.compile(
    r"crypto|bitcoin|btc|ethereum|eth|solana|sol|defi|nft|blockchain"
    r"|dogecoin|doge|xrp|ripple|cardano|polkadot|avalanche|chainlink|bnb|binance"
    r"|coinbase|stablecoin|memecoin|token", re.IGNORECASE,
)

POSITIVE_WORDS = frozenset([
    "win", "pass", "above", "exceed", "achieve", "surge", "gain", "rise",
    "increase", "approve", "success", "agree", "accept", "hit", "reach",
])
NEGATIVE_WORDS = frozenset([
    "lose", "fail", "below", "crash", "reject", "decline", "fall", "drop",
    "decrease", "deny", "miss", "ban", "block", "cancel", "collapse",
])
CERTAINTY_WORDS = frozenset([
    "will", "definitely", "certainly", "must", "always", "guaranteed",
])
MONTHS = frozenset([
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
])


def nlp_features(question: str) -> dict:
    q = question.strip()
    ql = q.lower()
    words = ql.split()
    n = max(len(words), 1)
    unique = set(words)
    return {
        "q_length": float(len(q)),
        "q_word_count": float(n),
        "q_avg_word_len": sum(len(w) for w in words) / n,
        "q_word_diversity": len(unique) / n,
        "q_has_number": float(bool(re.search(r"\d", q))),
        "q_has_year": float("202" in q),
        "q_has_percent": float("%" in q),
        "q_has_dollar": float("$" in q),
        "q_has_date": float(any(m in ql for m in MONTHS) or "/" in q),
        "q_starts_will": float(ql.startswith("will ")),
        "q_has_by": float(" by " in ql),
        "q_has_before": float(" before " in ql),
        "q_has_above": float(bool(re.search(r"above|over|exceed|hit|reach|break", ql))),
        "q_sentiment_pos": float(sum(1 for w in words if w in POSITIVE_WORDS)),
        "q_sentiment_neg": float(sum(1 for w in words if w in NEGATIVE_WORDS)),
        "q_certainty": float(sum(1 for w in words if w in CERTAINTY_WORDS)),
        "is_crypto": float(bool(CRYPTO_RE.search(q))),
        "is_sports": float(bool(SPORTS_RE.search(q))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build training data from Jon-Becker dataset")
    parser.add_argument("--data-dir", required=True, help="Path to polymarket data dir")
    parser.add_argument("--output", default="training_data_v5.json")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Max training samples to generate")
    parser.add_argument("--max-per-market", type=int, default=5,
                        help="Max samples per resolved market")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    markets_dir = data_dir / "markets"
    trades_dir = data_dir / "trades"

    # ---------------------------------------------------------------
    # Step 1: Resolve markets (all in-memory, ~100MB)
    # ---------------------------------------------------------------
    print("Step 1: Loading resolved markets...")
    con = duckdb.connect()

    markets_df = con.execute(f"""
        SELECT id, question, clob_token_ids, outcome_prices,
               volume, liquidity, end_date, created_at
        FROM '{markets_dir}/*.parquet'
        WHERE closed = true
    """).df()

    # Build token_id -> (won, market_info) mapping
    token_map = {}  # token_id -> (won: bool, info: dict)
    n_resolved = 0

    for _, row in markets_df.iterrows():
        try:
            prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
            if not prices or len(prices) != 2:
                continue
            p0, p1 = float(prices[0]), float(prices[1])
            if p0 > 0.99 and p1 < 0.01:
                winner = 0
            elif p0 < 0.01 and p1 > 0.99:
                winner = 1
            else:
                continue

            tokens = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
            if not tokens or len(tokens) != 2:
                continue

            info = {
                "question": row["question"] or "",
                "volume": float(row["volume"] or 0),
                "liquidity": float(row["liquidity"] or 0),
                "end_date": str(row["end_date"]) if row["end_date"] else None,
                "created_at": str(row["created_at"]) if row["created_at"] else None,
            }
            token_map[tokens[0]] = (winner == 0, info)
            token_map[tokens[1]] = (winner == 1, info)
            n_resolved += 1
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    print(f"  {n_resolved:,} resolved markets, {len(token_map):,} tokens")
    con.close()

    # Pre-compute NLP features per unique question
    print("  Pre-computing NLP features...")
    nlp_cache = {}
    for _, (_, info) in token_map.items():
        q = info["question"]
        if q not in nlp_cache:
            nlp_cache[q] = nlp_features(q)
    print(f"  {len(nlp_cache):,} unique questions")

    # ---------------------------------------------------------------
    # Step 2: Process trade files one at a time (memory-efficient)
    # ---------------------------------------------------------------
    print(f"\nStep 2: Processing trades (target {args.max_samples:,} samples)...")

    trade_files = sorted(trades_dir.glob("*.parquet"))
    snapshots = []
    token_counts = {}  # token_id -> count of samples taken
    files_ok = 0
    files_skip = 0
    trades_seen = 0

    for fi, tf in enumerate(trade_files):
        if len(snapshots) >= args.max_samples:
            break

        try:
            con = duckdb.connect()
            # Read one file, join with resolved tokens, sample
            df = con.execute(f"""
                SELECT
                    CASE
                        WHEN maker_asset_id = '0' THEN CAST(taker_asset_id AS VARCHAR)
                        ELSE CAST(maker_asset_id AS VARCHAR)
                    END AS token_id,
                    CASE
                        WHEN maker_asset_id = '0' THEN 1.0 * maker_amount / NULLIF(taker_amount, 0)
                        ELSE 1.0 * taker_amount / NULLIF(maker_amount, 0)
                    END AS price
                FROM '{tf}'
                WHERE taker_amount > 0 AND maker_amount > 0
            """).df()
            con.close()
        except Exception:
            files_skip += 1
            continue

        files_ok += 1
        trades_seen += len(df)

        # Filter to resolved tokens and valid prices
        df = df[df["token_id"].isin(token_map)]
        df = df[(df["price"] > 0.02) & (df["price"] < 0.98)]

        if df.empty:
            continue

        # Sample from this batch
        for token_id, grp in df.groupby("token_id"):
            already = token_counts.get(token_id, 0)
            remaining = args.max_per_market - already
            if remaining <= 0:
                continue

            sample = grp.sample(n=min(len(grp), remaining), random_state=42)
            won, info = token_map[token_id]
            nlp = nlp_cache[info["question"]]

            # Parse dates
            end_dt = None
            created_dt = None
            try:
                if info["end_date"] and info["end_date"] != "None":
                    end_dt = datetime.fromisoformat(str(info["end_date"]).replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, TypeError):
                pass
            try:
                if info["created_at"] and info["created_at"] != "None":
                    created_dt = datetime.fromisoformat(str(info["created_at"]).replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, TypeError):
                pass

            # Temporal features (use mid-point estimate since we lack exact trade timestamps)
            if end_dt and created_dt:
                total_span = (end_dt - created_dt).total_seconds() / 86400
                # Estimate: trades happen throughout market life. Use random offset.
                days_to_expiry = max(total_span * 0.3, 0.5)  # rough: ~70% through market life
                days_since_created = max(total_span * 0.7, 0.1)
                created_to_expiry_span = max(total_span, 0.1)
            else:
                days_to_expiry = 7.0
                days_since_created = 30.0
                created_to_expiry_span = 37.0

            # Skip if outside deployment window
            if days_to_expiry > 14.0:
                days_to_expiry = min(days_to_expiry, 14.0)

            log_volume = math.log1p(info["volume"])

            for _, trade in sample.iterrows():
                price = float(trade["price"])

                snap = {
                    "yes_price": price,
                    "momentum_1h": 0.0,  # not available from static trade data
                    "momentum_24h": 0.0,
                    "volatility_24h": 0.0,
                    "rsi": 0.5,
                    "log_volume": log_volume,
                    "days_to_expiry": days_to_expiry,
                    "is_crypto": nlp["is_crypto"],
                    "price_change_1d": 0.0,
                    "price_change_1w": 0.0,
                    "days_since_created": days_since_created,
                    "created_to_expiry_span": created_to_expiry_span,
                    "is_sports": nlp["is_sports"],
                    "q_length": nlp["q_length"],
                    "q_word_count": nlp["q_word_count"],
                    "q_avg_word_len": nlp["q_avg_word_len"],
                    "q_word_diversity": nlp["q_word_diversity"],
                    "q_has_number": nlp["q_has_number"],
                    "q_has_year": nlp["q_has_year"],
                    "q_has_percent": nlp["q_has_percent"],
                    "q_has_dollar": nlp["q_has_dollar"],
                    "q_has_date": nlp["q_has_date"],
                    "q_starts_will": nlp["q_starts_will"],
                    "q_has_by": nlp["q_has_by"],
                    "q_has_before": nlp["q_has_before"],
                    "q_has_above": nlp["q_has_above"],
                    "q_sentiment_pos": nlp["q_sentiment_pos"],
                    "q_sentiment_neg": nlp["q_sentiment_neg"],
                    "q_certainty": nlp["q_certainty"],
                    "outcome_yes": won,
                    "market_id": token_id,
                    "question": info["question"],
                    "snapshot_ts": 0,
                }
                snapshots.append(snap)
                token_counts[token_id] = token_counts.get(token_id, 0) + 1

        if args.verbose and (fi + 1) % 500 == 0:
            print(f"  [{fi+1}/{len(trade_files)}] {trades_seen:,} trades scanned, {len(snapshots):,} samples")

        if len(snapshots) >= args.max_samples:
            break

    print(f"  Files: {files_ok:,} ok, {files_skip:,} skipped")
    print(f"  Trades scanned: {trades_seen:,}")
    print(f"  Unique tokens sampled: {len(token_counts):,}")

    # ---------------------------------------------------------------
    # Step 3: Save
    # ---------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "source": "jon-becker-dataset",
            "n_snapshots": len(snapshots),
            "snapshots": snapshots,
        }, f)

    outcomes = [s["outcome_yes"] for s in snapshots]
    yes_pct = sum(outcomes) / len(outcomes) * 100 if outcomes else 0
    sports = sum(1 for s in snapshots if s["is_sports"] > 0.5)
    crypto = sum(1 for s in snapshots if s["is_crypto"] > 0.5)
    print(f"\nSaved {len(snapshots):,} snapshots to {output_path}")
    print(f"  Class balance: {yes_pct:.1f}% YES / {100-yes_pct:.1f}% NO")
    print(f"  Sports: {sports:,} ({100*sports/len(snapshots):.1f}%)")
    print(f"  Crypto: {crypto:,} ({100*crypto/len(snapshots):.1f}%)")


if __name__ == "__main__":
    main()
