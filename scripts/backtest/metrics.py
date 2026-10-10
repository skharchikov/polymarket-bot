"""Honest backtest metrics: net ROI, Brier skill vs market, drawdown, baseline."""

import numpy as np


def brier(probs, outcomes):
    p, o = np.asarray(probs, float), np.asarray(outcomes, float)
    return float(np.mean((p - o) ** 2))


def brier_skill_vs_market(model_probs, market_prices, outcomes):
    """1 - model_brier / market_brier. Negative means worse than the price."""
    mb = brier(model_probs, outcomes)
    kb = brier(market_prices, outcomes)
    return float(1.0 - mb / kb) if kb > 0 else 0.0


def max_drawdown(pnl_series):
    """Largest peak-to-trough drop of a cumulative PnL series."""
    s = np.asarray(pnl_series, float)
    if len(s) == 0:
        return 0.0
    peak = np.maximum.accumulate(s)
    return float(np.max(peak - s))


def summarize(bets):
    """bets: list of dicts with stake, entry_fee, pnl, won, model_prob,
    market_price, outcome. Returns an honest summary including net ROI on
    capital-at-risk (stake + entry fee) and a no-bet baseline."""
    if not bets:
        return {"n": 0, "net_pnl": 0.0, "net_roi_pct": 0.0}
    stake = sum(b["stake"] + b["entry_fee"] for b in bets)
    pnl = sum(b["pnl"] for b in bets)
    cum, run = [], 0.0
    for b in bets:
        run += b["pnl"]
        cum.append(run)
    return {
        "n": len(bets),
        "net_pnl": pnl,
        "net_roi_pct": (pnl / stake * 100.0) if stake > 0 else 0.0,
        "win_rate": sum(1 for b in bets if b["won"]) / len(bets),
        "brier_skill_vs_market": brier_skill_vs_market(
            [b["model_prob"] for b in bets],
            [b["market_price"] for b in bets],
            [b["outcome"] for b in bets],
        ),
        "max_drawdown": max_drawdown(cum),
        "baseline_no_bet_pnl": 0.0,
    }
