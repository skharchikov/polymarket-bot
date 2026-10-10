"""Leak-free walk-forward.

The scaler and the model are fit on the TRAIN slice of each fold only — never
on data that includes the test slice. This is the honesty fix for the old
`backtest.py`, which fit the RobustScaler on all rows before splitting.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def fit_scaler_on_train(x_train):
    """Fit a RobustScaler on the train slice only."""
    scaler = RobustScaler()
    scaler.fit(x_train)
    return scaler


@dataclass
class Fold:
    model: object
    x_test_scaled: pd.DataFrame
    y_test: np.ndarray
    prices_test: np.ndarray
    liquidity_test: np.ndarray


def market_grouped_splits(market_ids, snapshot_ts, n_splits=5):
    """Expanding-window folds grouped by market and ordered by first-seen time.

    All snapshots of a market land on ONE side of the split — critical because
    every snapshot of a market shares that market's final outcome as its label,
    so a snapshot-level (or time-level) split leaks the label across the same
    market's rows. Yields (train_mask, test_mask) boolean arrays.
    """
    mid = np.asarray(market_ids)
    ts = np.asarray(snapshot_ts)
    first_seen = {}
    for m, t in zip(mid, ts):
        if m not in first_seen or t < first_seen[m]:
            first_seen[m] = t
    markets = sorted(first_seen, key=lambda m: first_seen[m])
    n_m = len(markets)
    block = n_m // (n_splits + 1)
    if block == 0:
        return
    for k in range(1, n_splits + 1):
        train_markets = set(markets[: block * k])
        test_markets = set(markets[block * k : block * (k + 1)])
        if not test_markets:
            continue
        yield (
            np.array([m in train_markets for m in mid]),
            np.array([m in test_markets for m in mid]),
        )


def walk_forward(df, feature_cols, n_splits=5):
    """Yield market-grouped folds; scaler + calibrated model fit on train only."""
    # Lazy import so the scaler helper (and its unit test) don't require the
    # full ML stack.
    from sklearn.calibration import CalibratedClassifierCV

    from train_model import build_ensemble, build_feature_matrix

    x = build_feature_matrix(df).reset_index(drop=True)  # unscaled
    # Neutralize settled-delta features: in the training data (closed markets)
    # price_change_1d/1w come from the resolved market's Gamma field, so they
    # encode the outcome — a leak that inflates any historical eval. (These are
    # also dropped by the model-fixes plan.)
    for _leak in ("price_change_1d", "price_change_1w"):
        if _leak in x.columns:
            x[_leak] = 0.0
    y = np.asarray(df["label"].values)
    prices = np.asarray(df["yes_price"].values)
    # Liquidity proxy = market dollar volume (the `liquidity` column is unpopulated
    # in the training data). NaN volumes fall back to the median so they aren't
    # silently dropped. Real order-book depth is the S2 spike.
    if "volume" in df:
        vol = df["volume"].astype(float)
        liq = vol.fillna(vol.median()).clip(lower=0.0).values
    else:
        liq = np.full(len(df), 1e4)

    market_ids = df["market_id"].values
    snapshot_ts = df["snapshot_ts"].values if "snapshot_ts" in df else np.arange(len(df))

    for train_mask, test_mask in market_grouped_splits(market_ids, snapshot_ts, n_splits):
        x_train, x_test = x[train_mask], x[test_mask]
        scaler = fit_scaler_on_train(x_train)  # TRAIN ONLY
        x_tr = pd.DataFrame(scaler.transform(x_train), columns=feature_cols)
        x_te = pd.DataFrame(scaler.transform(x_test), columns=feature_cols)
        raw = build_ensemble()
        n_train = int(train_mask.sum())
        method = "isotonic" if n_train > 1000 else "sigmoid"
        cv = min(3, max(2, n_train // 200))
        model = CalibratedClassifierCV(raw, method=method, cv=cv, ensemble=True)
        model.fit(x_tr, y[train_mask])
        yield Fold(
            model=model,
            x_test_scaled=x_te,
            y_test=y[test_mask],
            prices_test=prices[test_mask],
            liquidity_test=liq[test_mask],
        )
