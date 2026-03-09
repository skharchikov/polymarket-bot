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
):
    """Simulate betting with a strategy profile.

    If use_bayesian=True, posteriors are already Bayesian-anchored.
    If use_bayesian=False, posteriors are raw model probs (old behavior).
    """
    s = STRATEGIES[strategy_name]
    bankroll_start = 300.0
    bankroll = bankroll_start
    flat_bankroll = 300.0  # flat sizing base (Rust bot uses flat, not compounding)
    pnl = 0.0
    n_bets = 0
    wins = 0
    losses = 0
    bets_log = []

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

        # XGBoost signals get halved thresholds (matches Rust)
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
        # Cap stake at remaining bankroll
        stake = min(stake, bankroll)
        if stake < 0.50:
            continue

        # Fees (3% slippage + trading)
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
        bets_log.append({
            "side": side,
            "price": bet_price,
            "prob": bet_prob,
            "edge": edge,
            "eff_edge": eff_edge,
            "conf": conf,
            "kelly": k,
            "stake": stake,
            "won": won,
            "pnl_after": pnl,
        })

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


def main():
    parser = argparse.ArgumentParser(description="Backtest with Bayesian anchoring")
    parser.add_argument("--input", default="model/training_data.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sweep", action="store_true", help="Sweep conservative params")
    args = parser.parse_args()

    df = load_data(args.input)
    if len(df) < 100:
        print(f"Need at least 100 samples, got {len(df)}", file=sys.stderr)
        sys.exit(1)

    if args.sweep:
        print(f"Sweeping conservative params on {len(df)} samples, {args.folds} folds")
        sweep_conservative(df, n_splits=args.folds)
    else:
        print(f"Running backtest on {len(df)} samples across {args.folds} folds")
        print(f"Comparing: OLD (raw model prob) vs NEW (Bayesian-anchored)")
        run_backtest(df, n_splits=args.folds)


if __name__ == "__main__":
    main()
