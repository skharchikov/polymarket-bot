#!/usr/bin/env python3
"""
One-off backfill: populate missing event_slug values in the bets table
by querying the Gamma API for each market.

Usage:
    python scripts/backfill_event_slugs.py [--db DATABASE_URL] [--dry-run]
"""

import argparse
import os
import time

import psycopg2
import requests

GAMMA_URL = "https://gamma-api.polymarket.com"


def get_event_slug(market_id: str) -> str | None:
    """Fetch event slug from Gamma API for a given market ID."""
    try:
        resp = requests.get(f"{GAMMA_URL}/markets", params={"id": market_id}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            data = data[0] if data else {}
        events = data.get("events", [])
        if events:
            return events[0].get("slug")
    except Exception as e:
        print(f"  [!] Failed to fetch market {market_id}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Backfill missing event_slug values")
    parser.add_argument("--db", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    if not args.db:
        print("ERROR: pass --db or set DATABASE_URL")
        return

    conn = psycopg2.connect(args.db)
    cur = conn.cursor()

    cur.execute(
        "SELECT id, market_id, question FROM bets "
        "WHERE event_slug IS NULL OR event_slug = '' "
        "ORDER BY id"
    )
    rows = cur.fetchall()
    print(f"Found {len(rows)} bets with missing event_slug\n")

    updated = 0
    for bet_id, market_id, question in rows:
        slug = get_event_slug(market_id)
        status = slug or "(no slug in API)"
        print(f"  #{bet_id} market={market_id} -> {status}")
        print(f"       {question}")

        if slug and not args.dry_run:
            cur.execute(
                "UPDATE bets SET event_slug = %s WHERE id = %s",
                (slug, bet_id),
            )
            updated += 1

        time.sleep(0.2)  # be nice to the API

    if not args.dry_run:
        conn.commit()
        print(f"\nDone. Updated {updated}/{len(rows)} bets.")
    else:
        print(f"\n[dry-run] Would update {updated}/{len(rows)} bets.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
