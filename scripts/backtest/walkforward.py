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


def walk_forward(df, feature_cols, n_splits=5):
    """Yield time-ordered folds; scaler + calibrated model fit on train only."""
    # Lazy import so the scaler helper (and its unit test) don't require the
    # full ML stack.
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit

    from train_model import build_ensemble, build_feature_matrix

    x = build_feature_matrix(df)  # unscaled
    y = df["label"].values
    prices = df["yes_price"].values
    if "log_volume" in df:
        liq = np.expm1(df["log_volume"].values)
    else:
        liq = np.full(len(df), 1e4)

    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(x):
        scaler = fit_scaler_on_train(x.iloc[train_idx])  # TRAIN ONLY
        x_tr = pd.DataFrame(scaler.transform(x.iloc[train_idx]), columns=feature_cols)
        x_te = pd.DataFrame(scaler.transform(x.iloc[test_idx]), columns=feature_cols)
        raw = build_ensemble()
        method = "isotonic" if len(train_idx) > 1000 else "sigmoid"
        cv = min(3, max(2, len(train_idx) // 200))
        model = CalibratedClassifierCV(raw, method=method, cv=cv, ensemble=True)
        model.fit(x_tr, y[train_idx])
        yield Fold(
            model=model,
            x_test_scaled=x_te,
            y_test=y[test_idx],
            prices_test=prices[test_idx],
            liquidity_test=liq[test_idx],
        )
