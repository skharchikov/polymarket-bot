"""Signal diagnostics for the honest harness.

A) learning curve + permutation test — is a null result data-limited or
   signal-limited? Uses a FAST single model (LightGBM) so we can afford many
   fits. Skill = Brier skill vs the market price on a held-out market-grouped
   test fold.
B) FLB robustness — per-fold consistency + bootstrap CI + ex-outlier PnL for
   the recalibration strategy, so we know if its edge is real or a few wins.
"""

import numpy as np

from backtest.metrics import brier_skill_vs_market
from backtest.walkforward import market_grouped_splits

LEAK_COLS = ("price_change_1d", "price_change_1w")


def _prep(df, feature_cols):
    from train_model import build_feature_matrix
    x = build_feature_matrix(df).reset_index(drop=True)
    for c in LEAK_COLS:
        if c in x.columns:
            x[c] = 0.0
    y = np.asarray(df["label"].values, int)
    price = np.asarray(df["yes_price"].values, float)
    mid = df["market_id"].values
    ts = df["snapshot_ts"].values if "snapshot_ts" in df else np.arange(len(df))
    return x, y, price, mid, ts


def _fast_fit_predict(x_tr, y_tr, x_te):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, verbosity=-1)
    m.fit(x_tr, y_tr)
    return m.predict_proba(x_te)[:, 1]


def learning_and_permutation(df, feature_cols, fractions=(0.25, 0.5, 1.0),
                             n_perm=10, seed_base=0):
    """Returns (learning_curve, real_skill, perm_skills). Single leave-out
    market-grouped fold (last block = test)."""
    x, y, price, mid, ts = _prep(df, feature_cols)
    # take the LAST fold from a 4-way market split as the held-out test
    folds = list(market_grouped_splits(mid, ts, n_splits=4))
    train_mask, test_mask = folds[-1]
    xtr_all, ytr_all = x[train_mask], y[train_mask]
    xte, yte, pte = x[test_mask], y[test_mask], price[test_mask]

    # learning curve: train on the LAST fraction of train markets (time-ordered)
    tr_idx = np.nonzero(train_mask)[0]
    curve = {}
    for f in fractions:
        k = max(200, int(len(tr_idx) * f))
        sub = tr_idx[-k:]
        probs = _fast_fit_predict(x.loc[sub], y[sub], xte)
        curve[f] = round(brier_skill_vs_market(probs, pte, yte), 4)

    # full-data real skill
    real = curve[fractions[-1]]

    # permutation null: shuffle train labels (market-level indices already mixed
    # in train; a plain shuffle breaks any feature->label link)
    perm = []
    for j in range(n_perm):
        rng = np.random.RandomState(seed_base + j + 1)
        yperm = ytr_all.copy()
        rng.shuffle(yperm)
        probs = _fast_fit_predict(xtr_all, yperm, xte)
        perm.append(round(brier_skill_vs_market(probs, pte, yte), 4))
    return curve, real, perm


def flb_robustness(pnls_list, n_boot=1000):
    """Bootstrap CI + ex-top-k for a strategy's per-bet PnL list."""
    pnls = np.array(sorted(pnls_list, reverse=True))
    total = float(pnls.sum())
    ex5 = float(pnls[5:].sum()) if len(pnls) > 5 else 0.0
    ex20 = float(pnls[20:].sum()) if len(pnls) > 20 else 0.0

    # bootstrap CI on total PnL (resample bets with replacement)
    rng = np.random.RandomState(0)
    boots = []
    n = len(pnls)
    if n > 0:
        for _ in range(n_boot):
            samp = pnls[rng.randint(0, n, n)]
            boots.append(samp.sum())
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = 0.0
    return {"n": n, "total": round(total, 1), "ex_top5": round(ex5, 1),
            "ex_top20": round(ex20, 1), "boot_ci95": (round(float(lo), 1), round(float(hi), 1))}
