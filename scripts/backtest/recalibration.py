"""Market-price recalibration (Le 2026): p* = p^θ / (p^θ + (1-p)^θ).

θ > 1 extremizes (market underconfident → push toward 0/1);
θ < 1 shrinks toward 0.5 (market overconfident). θ is fit per
(category, horizon) bucket by minimizing log-loss on TRAIN outcomes only.
"""

import numpy as np


def recalibrate(p, theta):
    """Apply the power/odds recalibration. Vectorized; clamps p away from 0/1."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    pt = p ** theta
    return pt / (pt + (1 - p) ** theta)


def _logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_theta(prices, outcomes, lo=0.3, hi=3.0, steps=54):
    """Fit θ minimizing recalibrated log-loss on (prices, outcomes).

    Returns 1.0 (no-op) when there is too little data to fit reliably.
    Uses a coarse grid + local refine — no scipy dependency.
    """
    prices = np.asarray(prices, float)
    outcomes = np.asarray(outcomes, float)
    if len(prices) < 50:
        return 1.0
    grid = np.linspace(lo, hi, steps)
    losses = [_logloss(recalibrate(prices, t), outcomes) for t in grid]
    best = grid[int(np.argmin(losses))]
    # local refine around the grid winner
    fine = np.linspace(max(lo, best - 0.1), min(hi, best + 0.1), 21)
    losses_f = [_logloss(recalibrate(prices, t), outcomes) for t in fine]
    return float(fine[int(np.argmin(losses_f))])


def category_bucket(cat):
    """Coarse category from the raw snapshot `category`/`question` string."""
    c = (cat or "").lower()
    if any(k in c for k in ("crypto", "bitcoin", "btc", "eth", "solana")):
        return "crypto"
    if any(k in c for k in ("nba", "nfl", "mlb", "nhl", "soccer", "sport", "ufc", "tennis", "game")):
        return "sports"
    if any(k in c for k in ("elect", "president", "senate", "poll", "trump", "politic")):
        return "politics"
    return "other"


def horizon_bucket(days_to_expiry):
    d = float(days_to_expiry)
    if d < 3:
        return "<3d"
    if d < 7:
        return "3-7d"
    return "7-14d"
