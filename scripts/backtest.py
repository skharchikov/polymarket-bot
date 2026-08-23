#!/usr/bin/env python3
"""
Backtest: simulate the full bot pipeline on historical data.

Matches the Rust bot logic:
1. Model predicts probability
2. Compute LR = model_prob / market_price (in odds space)
3. Dampen LR by confidence: LR^confidence
4. Bayesian update: posterior_odds = prior_odds * dampened_LR
5. Compute edge, apply strategy gates, Kelly sizing
6. Simulate P&L with flat bankroll

Usage:
    python scripts/backtest.py [--input model/training_data.json]
"""

import argparse
import json
import math
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*feature names.*")
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

from train_model import FEATURE_COLS, build_ensemble, build_feature_matrix, load_data

try:
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# --- Bayesian logic (mirrors Rust bayesian.rs) ---

def prob_to_odds(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return p / (1.0 - p)


def odds_to_prob(odds: float) -> float:
    if odds <= 0:
        return 0.001
    return max(0.001, min(0.999, odds / (1.0 + odds)))


def dampen_lr(lr: float, confidence: float) -> float:
    confidence = max(0.0, min(1.0, confidence))
    if lr <= 0:
        return 1.0
    return lr ** confidence


def bayesian_posterior(prior: float, lr: float, confidence: float) -> float:
    """Market-anchored posterior: prior + dampened model LR."""
    d_lr = dampen_lr(lr, confidence)
    prior_odds = prob_to_odds(prior)
    return odds_to_prob(prior_odds * d_lr)


def compute_lr(model_prob: float, market_price: float) -> float:
    """Likelihood ratio from model vs market."""
    if market_price <= 0.01 or market_price >= 0.99:
        return 1.0
    # Clamp the model prob away from 0/1 to avoid div-by-zero when the
    # calibrated model outputs an extreme probability.
    model_prob = max(0.001, min(0.999, model_prob))
    return (model_prob / market_price) / ((1.0 - model_prob) / (1.0 - market_price))


# --- Confidence estimation (mirrors serve_model.py) ---

def estimate_confidence(model, features_row):
    """Estimate confidence from base model disagreement."""
    try:
        # Wrap in DataFrame to suppress feature name warnings
        if not isinstance(features_row, pd.DataFrame):
            features_row = pd.DataFrame(features_row, columns=FEATURE_COLS)

        estimator = model
        if hasattr(estimator, "calibrated_classifiers_"):
            estimator = estimator.calibrated_classifiers_[0].estimator

        base_preds = []
        if hasattr(estimator, "estimators_"):
            for est_list in estimator.estimators_:
                if isinstance(est_list, list):
                    for est in est_list:
                        pred = est.predict_proba(features_row)[0, 1]
                        base_preds.append(pred)
                else:
                    pred = est_list.predict_proba(features_row)[0, 1]
                    base_preds.append(pred)

        if len(base_preds) >= 2:
            spread = max(base_preds) - min(base_preds)
            confidence = 0.75 / (1.0 + spread * 4.0)
            return max(0.25, min(0.75, confidence))
    except Exception:
        pass
    return 0.50


# --- Kelly criterion (mirrors Rust pricing/kelly.rs) ---

def kelly_fraction(prob: float, price: float) -> float:
    if price <= 0.0 or price >= 1.0 or prob <= 0.0 or prob >= 1.0:
        return 0.0
    b = (1.0 - price) / price
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(0.0, f)


def fractional_kelly(prob: float, price: float, fraction: float) -> float:
    return kelly_fraction(prob, price) * fraction


# --- Strategy profiles (mirrors Rust strategy.rs) ---

STRATEGIES = {
    "aggressive": {"kelly_frac": 0.50, "min_eff_edge": 0.025, "min_conf": 0.40},
    "balanced":   {"kelly_frac": 0.25, "min_eff_edge": 0.04,  "min_conf": 0.50},
    "conservative": {"kelly_frac": 0.10, "min_eff_edge": 0.06, "min_conf": 0.65},
}


def simulate_strategy(
    posteriors, confidences, prices, outcomes,
    strategy_name="balanced",
    use_bayesian=True,
    model_probs=None,
    stop_loss_pct=None,
    volatilities=None,
):
    """Simulate betting with a strategy profile.

    If stop_loss_pct is set, losing bets exit at stop-loss instead of full loss.
    Winning bets in volatile markets may get stopped out prematurely.
    """
    s = STRATEGIES[strategy_name]
    bankroll_start = 300.0
    bankroll = bankroll_start
    flat_bankroll = 300.0
    pnl = 0.0
    n_bets = 0
    wins = 0
    losses = 0
    stopped_out_winners = 0
    bets_log = []

    rng = np.random.RandomState(42)

    for i in range(len(posteriors)):
        post = posteriors[i]
        conf = confidences[i]
        price = prices[i]
        outcome = outcomes[i]

        if price <= 0.01 or price >= 0.99:
            continue

        # Determine best side
        yes_edge = post - price
        no_price = 1.0 - price
        no_prob = 1.0 - post
        no_edge = no_prob - no_price

        if yes_edge >= no_edge and yes_edge > 0:
            side = "YES"
            edge = yes_edge
            bet_price = price
            bet_prob = post
            won = outcome == 1
        elif no_edge > 0:
            side = "NO"
            edge = no_edge
            bet_price = no_price
            bet_prob = no_prob
            won = outcome == 0
        else:
            continue

        # Effective edge gate
        eff_edge = edge * conf
        min_edge = s["min_eff_edge"] * 0.5
        min_conf = s["min_conf"] * 0.7

        if eff_edge < min_edge:
            continue
        if conf < min_conf:
            continue

        # Kelly sizing
        k = fractional_kelly(bet_prob, bet_price, s["kelly_frac"])
        if k < 0.01:
            continue

        stake = flat_bankroll * k
        stake = min(stake, bankroll)
        if stake < 0.50:
            continue

        entry_fee = stake * 0.02
        exit_fee_rate = 0.02

        # --- Stop-loss logic ---
        if stop_loss_pct is not None:
            vol = volatilities[i] if volatilities is not None else 0.0

            if won:
                # Check if volatile enough to trigger premature stop-out.
                # Model: if 24h volatility > stop_loss / 2, there's a chance
                # the price whipsawed through our stop before recovering.
                # P(stop-out) = min(vol / stop_loss_pct, 0.5)
                p_stopped = min(vol / stop_loss_pct, 0.5) if stop_loss_pct > 0 else 0
                if rng.random() < p_stopped:
                    # Stopped out of a winner — lose stop_loss_pct of stake
                    exit_value = stake * (1.0 - stop_loss_pct)
                    exit_fee = exit_value * exit_fee_rate
                    loss = stake - exit_value + entry_fee + exit_fee
                    pnl -= loss
                    bankroll -= loss
                    losses += 1
                    stopped_out_winners += 1
                    n_bets += 1
                    bets_log.append({"won": False, "stopped": True, "pnl_after": pnl})
                    if bankroll <= 0:
                        break
                    continue
                else:
                    # Normal win
                    gross = stake * (1.0 - bet_price) / bet_price
                    exit_fee = gross * exit_fee_rate
                    profit = gross - entry_fee - exit_fee
                    pnl += profit
                    bankroll += profit
                    wins += 1
            else:
                # Losing bet — stop-loss caps the loss
                exit_value = stake * (1.0 - stop_loss_pct)
                exit_fee = exit_value * exit_fee_rate
                loss = stake - exit_value + entry_fee + exit_fee
                pnl -= loss
                bankroll -= loss
                losses += 1
        else:
            # No stop-loss — original behavior
            fee = stake * 0.03
            if won:
                profit = stake * (1.0 - bet_price) / bet_price - fee
                pnl += profit
                bankroll += profit
                wins += 1
            else:
                loss = stake + fee
                pnl -= loss
                bankroll -= loss
                losses += 1

        n_bets += 1
        bets_log.append({"won": won, "stopped": False, "pnl_after": pnl})

        if bankroll <= 0:
            break

    return {
        "strategy": strategy_name,
        "use_bayesian": use_bayesian,
        "n_bets": n_bets,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(n_bets, 1),
        "pnl": pnl,
        "bankroll_start": bankroll_start,
        "bankroll_end": bankroll,
        "roi": pnl / bankroll_start * 100,
        "stopped_out_winners": stopped_out_winners,
        "bets_log": bets_log,
    }


def run_backtest(df, n_splits=5):
    """Run backtest with time-series CV, comparing old vs new logic."""
    X = build_feature_matrix(df)
    y = df["label"].values
    prices = df["yes_price"].values

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_results = {"old": [], "new": []}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        p_test = prices[test_idx]

        # Build and train ensemble
        raw_model = build_ensemble()
        n_train = len(X_train)
        cal_method = "isotonic" if n_train > 1000 else "sigmoid"
        cal_cv = min(3, max(2, n_train // 200))

        model = CalibratedClassifierCV(
            estimator=raw_model,
            method=cal_method,
            cv=cal_cv,
            ensemble=True,
        )
        model.fit(X_train, y_train)

        # Get model predictions
        model_probs = model.predict_proba(X_test)[:, 1]

        # Estimate confidence for each sample
        confs = []
        for i in range(len(X_test)):
            conf = estimate_confidence(model, X_test[i:i+1])
            confs.append(conf)
        confs = np.array(confs)

        # --- OLD behavior: raw model prob as posterior ---
        old_posteriors = model_probs.copy()

        # --- NEW behavior: Bayesian anchoring ---
        new_posteriors = []
        for prob, price, conf in zip(model_probs, p_test, confs):
            lr = compute_lr(prob, price)
            post = bayesian_posterior(price, lr, conf)
            new_posteriors.append(post)
        new_posteriors = np.array(new_posteriors)

        print(f"\n{'='*60}")
        print(f"Fold {fold} ({len(test_idx)} test samples)")
        print(f"{'='*60}")

        # Show distribution of model predictions
        print(f"\n  Model prob range: [{model_probs.min():.2f}, {model_probs.max():.2f}]")
        print(f"  Market price range: [{p_test.min():.2f}, {p_test.max():.2f}]")
        print(f"  Confidence range: [{confs.min():.2f}, {confs.max():.2f}]")

        # Compare old vs new posteriors
        old_max_divergence = np.max(np.abs(old_posteriors - p_test))
        new_max_divergence = np.max(np.abs(new_posteriors - p_test))
        print(f"\n  Max divergence from market:")
        print(f"    Old (raw model):     {old_max_divergence:.1%}")
        print(f"    New (Bayes-anchored): {new_max_divergence:.1%}")

        for strategy in ["aggressive", "balanced", "conservative"]:
            old_res = simulate_strategy(
                old_posteriors, confs, p_test, y_test,
                strategy_name=strategy, use_bayesian=False, model_probs=model_probs
            )
            new_res = simulate_strategy(
                new_posteriors, confs, p_test, y_test,
                strategy_name=strategy, use_bayesian=True, model_probs=model_probs
            )

            all_results["old"].append(old_res)
            all_results["new"].append(new_res)

            print(f"\n  [{strategy.upper()}]")
            print(f"    OLD: {old_res['n_bets']:3d} bets, "
                  f"W/L {old_res['wins']}/{old_res['losses']}, "
                  f"WR {old_res['win_rate']:.0%}, "
                  f"PnL €{old_res['pnl']:+.2f}, "
                  f"ROI {old_res['roi']:+.1f}%")
            print(f"    NEW: {new_res['n_bets']:3d} bets, "
                  f"W/L {new_res['wins']}/{new_res['losses']}, "
                  f"WR {new_res['win_rate']:.0%}, "
                  f"PnL €{new_res['pnl']:+.2f}, "
                  f"ROI {new_res['roi']:+.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*60}")

    for strategy in ["aggressive", "balanced", "conservative"]:
        old_strat = [r for r in all_results["old"] if r["strategy"] == strategy]
        new_strat = [r for r in all_results["new"] if r["strategy"] == strategy]

        old_pnl = sum(r["pnl"] for r in old_strat)
        new_pnl = sum(r["pnl"] for r in new_strat)
        old_bets = sum(r["n_bets"] for r in old_strat)
        new_bets = sum(r["n_bets"] for r in new_strat)
        old_wins = sum(r["wins"] for r in old_strat)
        new_wins = sum(r["wins"] for r in new_strat)
        old_wr = old_wins / max(old_bets, 1)
        new_wr = new_wins / max(new_bets, 1)

        print(f"\n  [{strategy.upper()}]")
        print(f"    OLD (raw model):      {old_bets:3d} bets, WR {old_wr:.0%}, "
              f"Total PnL €{old_pnl:+.2f}")
        print(f"    NEW (Bayes-anchored): {new_bets:3d} bets, WR {new_wr:.0%}, "
              f"Total PnL €{new_pnl:+.2f}")
        diff = new_pnl - old_pnl
        print(f"    Delta: €{diff:+.2f}")


def sweep_conservative(df, n_splits=5):
    """Sweep conservative strategy parameters to find optimal settings."""
    X = build_feature_matrix(df)
    y = df["label"].values
    prices = df["yes_price"].values

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Pre-compute fold data (expensive part)
    fold_data = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train = X_scaled.iloc[train_idx]
        X_test = X_scaled.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        p_test = prices[test_idx]

        raw_model = build_ensemble()
        n_train = len(X_train)
        cal_method = "isotonic" if n_train > 1000 else "sigmoid"
        cal_cv = min(3, max(2, n_train // 200))

        model = CalibratedClassifierCV(
            estimator=raw_model, method=cal_method, cv=cal_cv, ensemble=True,
        )
        model.fit(X_train, y_train)

        model_probs = model.predict_proba(X_test)[:, 1]
        confs = np.array([estimate_confidence(model, X_test.iloc[i:i+1]) for i in range(len(X_test))])

        posteriors = np.array([
            bayesian_posterior(price, compute_lr(prob, price), conf)
            for prob, price, conf in zip(model_probs, p_test, confs)
        ])

        fold_data.append((posteriors, confs, p_test, y_test))
        print(f"  Fold {fold} prepared ({len(test_idx)} samples)")

    # Parameter grid
    kelly_fracs = [0.10, 0.15, 0.20]
    min_edges = [0.06, 0.08, 0.10, 0.12]
    min_confs = [0.40, 0.45, 0.50, 0.55, 0.65]

    print(f"\n{'='*80}")
    print(f"CONSERVATIVE STRATEGY PARAMETER SWEEP")
    print(f"{'='*80}")
    print(f"{'Kelly':>6s} {'MinEdge':>8s} {'MinConf':>8s} | {'Bets':>5s} {'WR':>5s} {'PnL':>10s} {'ROI':>7s} | {'Avg/fold':>10s}")
    print(f"{'-'*80}")

    best = None
    results = []

    for kf in kelly_fracs:
        for me in min_edges:
            for mc in min_confs:
                # Override strategy params
                STRATEGIES["_sweep"] = {"kelly_frac": kf, "min_eff_edge": me, "min_conf": mc}

                total_pnl = 0
                total_bets = 0
                total_wins = 0

                for posteriors, confs, p_test, y_test in fold_data:
                    res = simulate_strategy(
                        posteriors, confs, p_test, y_test,
                        strategy_name="_sweep", use_bayesian=True,
                    )
                    total_pnl += res["pnl"]
                    total_bets += res["n_bets"]
                    total_wins += res["wins"]

                wr = total_wins / max(total_bets, 1)
                roi = total_pnl / (300.0 * n_splits) * 100
                avg_fold = total_pnl / n_splits

                row = {
                    "kelly": kf, "min_edge": me, "min_conf": mc,
                    "bets": total_bets, "wr": wr, "pnl": total_pnl,
                    "roi": roi, "avg_fold": avg_fold,
                }
                results.append(row)

                if total_bets >= 10:  # need enough bets for statistical relevance
                    print(f"{kf:>6.2f} {me:>8.2f} {mc:>8.2f} | "
                          f"{total_bets:>5d} {wr:>4.0%} {total_pnl:>+10.2f} {roi:>+6.1f}% | "
                          f"{avg_fold:>+10.2f}")

                    if best is None or (total_pnl > best["pnl"] and wr >= 0.45):
                        best = row

    # Clean up
    del STRATEGIES["_sweep"]

    if best:
        print(f"\n  BEST: kelly={best['kelly']:.2f}, min_edge={best['min_edge']:.2f}, "
              f"min_conf={best['min_conf']:.2f}")
        print(f"        {best['bets']} bets, WR {best['wr']:.0%}, "
              f"PnL €{best['pnl']:+.2f}, ROI {best['roi']:+.1f}%")

    return best


def sweep_stop_loss(df, n_splits=5):
    """Compare different stop-loss levels across all strategies."""
    X = build_feature_matrix(df)
    y = df["label"].values
    prices = df["yes_price"].values
    # volatility_24h is feature index 3 in FEATURE_COLS
    vol_idx = FEATURE_COLS.index("volatility_24h")

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Pre-compute fold data
    fold_data = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train = X_scaled.iloc[train_idx]
        X_test = X_scaled.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        p_test = prices[test_idx]
        # Use raw (unscaled) volatility for stop-loss simulation
        vol_test = X.iloc[test_idx]["volatility_24h"].values

        raw_model = build_ensemble()
        n_train = len(X_train)
        cal_method = "isotonic" if n_train > 1000 else "sigmoid"
        cal_cv = min(3, max(2, n_train // 200))

        model = CalibratedClassifierCV(
            estimator=raw_model, method=cal_method, cv=cal_cv, ensemble=True,
        )
        model.fit(X_train, y_train)

        model_probs = model.predict_proba(X_test)[:, 1]
        confs = np.array([estimate_confidence(model, X_test.iloc[i:i+1]) for i in range(len(X_test))])

        posteriors = np.array([
            bayesian_posterior(price, compute_lr(prob, price), conf)
            for prob, price, conf in zip(model_probs, p_test, confs)
        ])

        fold_data.append((posteriors, confs, p_test, y_test, vol_test))
        print(f"  Fold {fold} prepared ({len(test_idx)} samples, avg vol={vol_test.mean():.3f})")

    stop_loss_levels = [None, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    print(f"\n{'='*90}")
    print(f"STOP-LOSS COMPARISON (Bayesian-anchored, all folds combined)")
    print(f"{'='*90}")

    for strategy in ["aggressive", "balanced", "conservative"]:
        print(f"\n  [{strategy.upper()}]")
        print(f"  {'StopLoss':>10s} | {'Bets':>5s} {'WR':>5s} {'Stopped':>8s} {'PnL':>10s} {'ROI':>7s} | {'vs None':>10s}")
        print(f"  {'-'*70}")

        baseline_pnl = None

        for sl in stop_loss_levels:
            total_pnl = 0
            total_bets = 0
            total_wins = 0
            total_stopped = 0

            for posteriors, confs, p_test, y_test, vol_test in fold_data:
                res = simulate_strategy(
                    posteriors, confs, p_test, y_test,
                    strategy_name=strategy, use_bayesian=True,
                    stop_loss_pct=sl, volatilities=vol_test,
                )
                total_pnl += res["pnl"]
                total_bets += res["n_bets"]
                total_wins += res["wins"]
                total_stopped += res.get("stopped_out_winners", 0)

            wr = total_wins / max(total_bets, 1)
            roi = total_pnl / (300.0 * n_splits) * 100

            if sl is None:
                baseline_pnl = total_pnl
                sl_label = "None"
            else:
                sl_label = f"{sl:.0%}"

            delta = total_pnl - baseline_pnl if baseline_pnl is not None else 0

            print(f"  {sl_label:>10s} | "
                  f"{total_bets:>5d} {wr:>4.0%} {total_stopped:>8d} "
                  f"{total_pnl:>+10.2f} {roi:>+6.1f}% | "
                  f"{delta:>+10.2f}")


# --- Honest end-to-end backtest (leak-free, real fills + fees) ---

# Live decision constants (mirror trading-bot): LR is dampened by
# confidence * LR_DAMPING before the edge floor; YES bets are blocked.
LR_DAMPING = 0.5
EDGE_FLOOR = 0.02          # live.rs:1236 effective_edge (edge * conf) floor
MIN_KELLY = 0.02          # config.rs min_kelly_size
MIN_BET_PRICE = 0.15      # config.rs min_bet_price
BLOCK_YES_SIDE = True     # config.rs block_yes_side ("30% WR, -€745 in prod")
FLAT_BANKROLL = 300.0


def _batch_confidence(model, x_test):
    """Vectorized agreement-based confidence for a whole test matrix at once
    (same formula as serve_model._estimate_confidence, but one predict_proba
    per base estimator instead of per row)."""
    est = model
    if hasattr(est, "calibrated_classifiers_"):
        est = est.calibrated_classifiers_[0].estimator
    base = []
    if hasattr(est, "estimators_"):
        for e in est.estimators_:
            members = e if isinstance(e, list) else [e]
            for member in members:
                base.append(member.predict_proba(x_test)[:, 1])
    if len(base) < 2:
        return np.full(len(x_test), 0.50)
    base = np.asarray(base)
    spread = base.max(axis=0) - base.min(axis=0)
    return np.clip(0.75 / (1.0 + spread * 4.0), 0.25, 0.75)


def run_honest_backtest(df, n_splits=5, strategy="balanced",
                        block_yes=BLOCK_YES_SIDE, edge_floor=EDGE_FLOOR,
                        lr_damping=LR_DAMPING):
    """Leak-free, fee- and fill-aware simulation of the DEPLOYED decision.

    Reproduces live behaviour (LR dampening, gates incl. block_yes_side, the
    agreement-based confidence) so a profit here would be real.
    """
    cached = _materialize_folds(df, n_splits=n_splits)
    return _evaluate_folds(cached, strategy, block_yes, edge_floor, lr_damping)


def run_recalibration_backtest(df, n_splits=5, strategy="balanced", edge_floor=EDGE_FLOOR):
    """No-ML strategy: bet where a per-(category,horizon) recalibration of the
    MARKET PRICE diverges from the price. θ fit on train folds only (leak-free
    via the market-grouped split). Fast — no ensemble training."""
    from backtest.fees import net_pnl
    from backtest.fills import fill_price, max_fillable
    from backtest.metrics import summarize
    from backtest.recalibration import (category_bucket, fit_theta,
                                        horizon_bucket, recalibrate)
    from backtest.walkforward import market_grouped_splits

    s = STRATEGIES[strategy]
    price = df["yes_price"].astype(float).values
    outcome = df["label"].astype(int).values
    dte = df["days_to_expiry"].astype(float).values
    catcol = df["category"] if "category" in df else df["question"]
    cats = [category_bucket(str(c)) for c in catcol.values]
    bucket = np.array([f"{c}|{horizon_bucket(d)}" for c, d in zip(cats, dte)])
    vol = df["volume"].astype(float)
    liq = vol.fillna(vol.median()).clip(lower=0.0).values
    mid = df["market_id"].values
    ts = df["snapshot_ts"].values if "snapshot_ts" in df else np.arange(len(df))

    bets = []
    funnel = {"priced": 0, "no_edge": 0, "edge_floor": 0, "min_price": 0,
              "min_kelly": 0, "stake_too_small": 0, "placed": 0}
    theta_log = {}

    for train_mask, test_mask in market_grouped_splits(mid, ts, n_splits):
        thetas = {}
        for b in np.unique(bucket[train_mask]):
            m = train_mask & (bucket == b)
            thetas[b] = fit_theta(price[m], outcome[m])
            theta_log.setdefault(b, []).append(thetas[b])
        for i in np.nonzero(test_mask)[0]:
            p = float(price[i])
            if p <= 0.01 or p >= 0.99:
                continue
            funnel["priced"] += 1
            post = float(recalibrate(p, thetas.get(bucket[i], 1.0)))
            yes_edge = post - p
            no_edge = (1.0 - post) - (1.0 - p)
            if yes_edge >= no_edge and yes_edge > 0:
                side, edge, bet_price, bet_prob, won = "YES", yes_edge, p, post, outcome[i] == 1
            elif no_edge > 0:
                side, edge, bet_price, bet_prob, won = "NO", no_edge, 1.0 - p, 1.0 - post, outcome[i] == 0
            else:
                funnel["no_edge"] += 1
                continue
            if edge < edge_floor:
                funnel["edge_floor"] += 1
                continue
            if bet_price < MIN_BET_PRICE:
                funnel["min_price"] += 1
                continue
            k = fractional_kelly(bet_prob, bet_price, s["kelly_frac"])
            if k < MIN_KELLY:
                funnel["min_kelly"] += 1
                continue
            stake = min(FLAT_BANKROLL * k, max_fillable(liq[i]))
            if stake < 0.50:
                funnel["stake_too_small"] += 1
                continue
            funnel["placed"] += 1
            filled = fill_price(bet_price, stake, liq[i])
            pnl = net_pnl(stake, filled, won)
            bets.append({"stake": stake, "entry_fee": stake * 0.02, "pnl": pnl,
                         "won": won, "model_prob": post, "market_price": p,
                         "outcome": int(outcome[i])})

    out = summarize(bets)
    out["funnel"] = funnel
    out["theta_by_bucket"] = {b: round(float(np.mean(v)), 2) for b, v in sorted(theta_log.items())}
    out["pnls"] = [b["pnl"] for b in bets]
    return out


def _materialize_folds(df, n_splits=5):
    """Train the (expensive) per-fold models ONCE and cache each fold's
    predictions + test arrays, so many gate configs evaluate cheaply."""
    from backtest.walkforward import walk_forward
    cached = []
    for fold in walk_forward(df, FEATURE_COLS, n_splits=n_splits):
        cached.append({
            "probs": fold.model.predict_proba(fold.x_test_scaled)[:, 1],
            "confs": _batch_confidence(fold.model, fold.x_test_scaled),
            "prices": fold.prices_test,
            "outcomes": fold.y_test,
            "liq": fold.liquidity_test,
        })
    return cached


def _evaluate_folds(cached, strategy="balanced", block_yes=BLOCK_YES_SIDE,
                    edge_floor=EDGE_FLOOR, lr_damping=LR_DAMPING):
    """Replay the deployed decision over cached folds under one gate config."""
    from backtest.fees import net_pnl
    from backtest.fills import fill_price, max_fillable
    from backtest.metrics import summarize

    s = STRATEGIES[strategy]
    bets = []
    funnel = {"priced": 0, "no_edge": 0, "block_yes": 0, "edge_floor": 0,
              "min_price": 0, "min_kelly": 0, "stake_too_small": 0, "placed": 0}

    for fold in cached:
        probs, confs = fold["probs"], fold["confs"]
        for i in range(len(probs)):
            prob = float(probs[i])
            price = float(fold["prices"][i])
            outcome = int(fold["outcomes"][i])
            liq = float(fold["liq"][i])
            if price <= 0.01 or price >= 0.99:
                continue
            funnel["priced"] += 1

            conf = float(confs[i])
            lr = compute_lr(prob, price)
            post = bayesian_posterior(price, lr, conf * lr_damping)

            yes_edge = post - price
            no_edge = (1.0 - post) - (1.0 - price)
            if yes_edge >= no_edge and yes_edge > 0:
                side, edge, bet_price, bet_prob, won = "YES", yes_edge, price, post, outcome == 1
            elif no_edge > 0:
                side, edge, bet_price, bet_prob, won = "NO", no_edge, 1.0 - price, 1.0 - post, outcome == 0
            else:
                funnel["no_edge"] += 1
                continue

            if block_yes and side == "YES":
                funnel["block_yes"] += 1
                continue
            if edge * conf < edge_floor:
                funnel["edge_floor"] += 1
                continue
            if bet_price < MIN_BET_PRICE:
                funnel["min_price"] += 1
                continue

            k = fractional_kelly(bet_prob, bet_price, s["kelly_frac"])
            if k < MIN_KELLY:
                funnel["min_kelly"] += 1
                continue

            stake = min(FLAT_BANKROLL * k, max_fillable(liq))
            if stake < 0.50:
                funnel["stake_too_small"] += 1
                continue

            funnel["placed"] += 1
            filled = fill_price(bet_price, stake, liq)
            entry_fee = stake * 0.02
            pnl = net_pnl(stake, filled, won)
            bets.append({
                "stake": stake, "entry_fee": entry_fee, "pnl": pnl, "won": won,
                "model_prob": prob, "market_price": price, "outcome": outcome,
            })

    out = summarize(bets)
    out["funnel"] = funnel
    return out


def main():
    parser = argparse.ArgumentParser(description="Backtest with Bayesian anchoring")
    parser.add_argument("--input", default="model/training_data.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--honest", action="store_true",
                        help="Leak-free, fee/fill-aware honest backtest (default)")
    parser.add_argument("--sweep", action="store_true", help="Sweep conservative params (LEAKY — exploratory only)")
    parser.add_argument("--stop-loss", action="store_true", help="Compare stop-loss levels (LEAKY — exploratory only)")
    parser.add_argument("--legacy", action="store_true",
                        help="Old OLD-vs-NEW comparison (LEAKY scaler — exploratory only)")
    parser.add_argument("--landscape", action="store_true",
                        help="Train folds once, sweep gate configs (block_yes / edge_floor / lr_damping)")
    parser.add_argument("--recal", action="store_true",
                        help="No-ML market-price recalibration strategy (per category x horizon)")
    args = parser.parse_args()

    df = load_data(args.input)
    if len(df) < 100:
        print(f"Need at least 100 samples, got {len(df)}", file=sys.stderr)
        sys.exit(1)

    if args.sweep:
        print("WARNING: sweep uses pre-split scaling (look-ahead leakage) — exploratory only.\n")
        print(f"Sweeping conservative params on {len(df)} samples, {args.folds} folds")
        sweep_conservative(df, n_splits=args.folds)
    elif args.stop_loss:
        print("WARNING: stop-loss sweep uses pre-split scaling (look-ahead leakage) — exploratory only.\n")
        print(f"Comparing stop-loss levels on {len(df)} samples, {args.folds} folds")
        sweep_stop_loss(df, n_splits=args.folds)
    elif args.legacy:
        print("WARNING: legacy path fits the scaler on ALL rows (look-ahead leakage) — exploratory only.\n")
        run_backtest(df, n_splits=args.folds)
    elif args.recal:
        print(f"Recalibration backtest on {len(df)} samples, {args.folds} folds "
              f"(no ML; θ per category×horizon, fit on train only)")
        for strategy in ("aggressive", "balanced", "conservative"):
            r = run_recalibration_backtest(df, n_splits=args.folds, strategy=strategy)
            print(f"\n[{strategy.upper()}]")
            print(f"  bets={r['n']}  net_pnl=€{r.get('net_pnl', 0):+.2f}  "
                  f"net_roi={r.get('net_roi_pct', 0):+.1f}%  "
                  f"win_rate={r.get('win_rate', 0):.0%}  "
                  f"brier_skill_vs_market={r.get('brier_skill_vs_market', 0):+.2f}  "
                  f"max_dd=€{r.get('max_drawdown', 0):.2f}")
            if strategy == "balanced":
                print(f"  funnel: {r.get('funnel', {})}")
                print(f"  θ by bucket: {r.get('theta_by_bucket', {})}")
                pnls = sorted(r.get("pnls", []), reverse=True)
                if pnls:
                    total = sum(pnls)
                    top5 = sum(pnls[:5])
                    ex_top5 = total - top5
                    wins = [p for p in pnls if p > 0]
                    print(f"  concentration: total=€{total:+.0f}  top5_wins=€{top5:+.0f} "
                          f"({(top5 / total * 100 if total else 0):.0f}% of net)  "
                          f"net_excl_top5=€{ex_top5:+.0f}  n_wins={len(wins)}")
    elif args.landscape:
        print(f"Landscape sweep on {len(df)} samples, {args.folds} folds "
              f"(training once, replaying gate configs)")
        cached = _materialize_folds(df, n_splits=args.folds)
        configs = [
            ("base: block_yes, floor .02, damp .5", dict(block_yes=True, edge_floor=0.02, lr_damping=0.5)),
            ("allow_yes", dict(block_yes=False, edge_floor=0.02, lr_damping=0.5)),
            ("trust model (damp 1.0)", dict(block_yes=True, edge_floor=0.02, lr_damping=1.0)),
            ("trust market (damp 0.2)", dict(block_yes=True, edge_floor=0.02, lr_damping=0.2)),
            ("higher floor .05", dict(block_yes=True, edge_floor=0.05, lr_damping=0.5)),
            ("allow_yes + damp 1.0", dict(block_yes=False, edge_floor=0.02, lr_damping=1.0)),
            ("allow_yes + floor .05 + damp 1.0", dict(block_yes=False, edge_floor=0.05, lr_damping=1.0)),
        ]
        print(f"\n{'config':<40s} {'bets':>6s} {'net_pnl':>10s} {'roi%':>7s} {'win%':>5s} {'skill':>6s}")
        for label, cfg in configs:
            r = _evaluate_folds(cached, strategy="balanced", **cfg)
            print(f"{label:<40s} {r['n']:>6d} {r.get('net_pnl', 0):>+10.0f} "
                  f"{r.get('net_roi_pct', 0):>+6.1f}% {r.get('win_rate', 0)*100:>4.0f}% "
                  f"{r.get('brier_skill_vs_market', 0):>+6.2f}")
    else:
        print(f"Honest backtest on {len(df)} samples across {args.folds} folds")
        for strategy in ("aggressive", "balanced", "conservative"):
            r = run_honest_backtest(df, n_splits=args.folds, strategy=strategy)
            print(f"\n[{strategy.upper()}]")
            print(f"  bets={r['n']}  net_pnl=€{r.get('net_pnl', 0):+.2f}  "
                  f"net_roi={r.get('net_roi_pct', 0):+.1f}%  "
                  f"win_rate={r.get('win_rate', 0):.0%}  "
                  f"brier_skill_vs_market={r.get('brier_skill_vs_market', 0):+.2f}  "
                  f"max_dd=€{r.get('max_drawdown', 0):.2f}")
            print(f"  funnel: {r.get('funnel', {})}")


if __name__ == "__main__":
    main()
