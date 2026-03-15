#!/usr/bin/env python3
"""
Phase 2: Train XGBoost + LightGBM ensemble on Polymarket snapshot data.

Reads training_data.json from fetch_data.py, trains a stacking ensemble,
calibrates probabilities, and exports model as JSON for Rust inference.

Usage:
    python scripts/train_model.py [--input model/training_data.json] [--output model/xgb_model.json]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed, using sklearn only")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed, using sklearn only")


# Feature columns in fixed order — must match Rust inference (MarketFeatures::NAMES).
# news_count / best_news_score / avg_news_age_hours / order_imbalance / spread removed:
# they are always 0 in training data (not available retroactively) but non-zero in live
# inference, causing guaranteed distribution shift with zero learning signal.
FEATURE_COLS = [
    "yes_price",
    "momentum_1h",
    "momentum_24h",
    "volatility_24h",
    "rsi",
    "log_volume",
    "log_liquidity",
    "days_to_expiry",
    "is_crypto",
    "is_politics",
    "is_sports",
    "price_change_1d",
    "price_change_1w",
]

N_FEATURES = len(FEATURE_COLS)


def load_data(path: str) -> pd.DataFrame:
    """Load and prepare training data."""
    with open(path) as f:
        raw = json.load(f)

    snapshots = raw["snapshots"]
    print(f"Loaded {len(snapshots)} snapshots from {raw['n_markets']} markets")

    df = pd.DataFrame(snapshots)

    # Derived features
    df["log_volume"] = np.log1p(df["volume"])
    df["log_liquidity"] = np.log1p(df["liquidity"])

    # Category one-hot — keywords must match live Rust logic in features.rs exactly.
    # fetch_data.py stores combined "category + question" in the category field.
    combined = df["category"].fillna("").str.lower()
    df["is_crypto"] = combined.str.contains(
        "crypto|bitcoin|btc|ethereum|eth|solana|sol|defi|nft|blockchain"
        "|dogecoin|doge|xrp|ripple|cardano|polkadot|avalanche|chainlink|bnb|binance"
        "|coinbase|stablecoin|memecoin|token"
    ).astype(float)
    df["is_politics"] = combined.str.contains(
        "politic|election|president|vote|congress|senate|democrat|republican"
    ).astype(float)
    df["is_sports"] = combined.str.contains(
        "sport|nba|nfl|soccer|tennis|mma|mlb|nhl|football|basketball|baseball"
    ).astype(float)

    # Fill NaN momentum with 0 (no price reference available)
    for col in ["momentum_1h", "momentum_24h", "price_1h_ago", "price_6h_ago", "price_24h_ago"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Sort by snapshot time for proper time-series splits
    df = df.sort_values("snapshot_ts").reset_index(drop=True)

    # Label
    df["label"] = df["outcome_yes"].astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract feature matrix in fixed column order as a DataFrame."""
    # Fill missing columns with 0 (backward compat with old training data)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    X = df[FEATURE_COLS].astype(np.float64).copy()
    X = X.fillna(0.0).replace([np.inf, -np.inf], [1.0, 0.0])
    return X


def build_ensemble():
    """Build stacking ensemble from multiple boosting models."""
    base = []

    if HAS_XGBOOST:
        base.append(("xgb", xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )))

    if HAS_LIGHTGBM:
        base.append(("lgb", lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary",
            metric="binary_logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )))

    base.append(("hgb", HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=5,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )))

    base.append(("et", ExtraTreesClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )))

    base.append(("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )))

    stacker = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    return stacker


def evaluate_folds(model, X, y, prices, n_splits=5, market_ids=None):
    """Time-series cross-validation with market-relevant metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        p_test = prices[test_idx]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        brier = brier_score_loss(y_test, probs)
        ll = log_loss(y_test, probs)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # The real test: edge vs market price
        edges = probs - p_test
        avg_edge = float(np.mean(edges))

        # Simulated P&L using flat Kelly sizing (matches Rust bot)
        # Rust bot uses flat €300 bankroll per strategy, not compounding.
        # Kelly: f = (b*p - q) / b  where b = (1-price)/price
        kelly_fraction = 0.25  # quarter-Kelly, same as balanced strategy
        bankroll = 300.0  # flat bankroll — does NOT compound
        pnl = 0.0
        n_bets = 0
        for prob, price, outcome in zip(probs, p_test, y_test):
            if price <= 0.01 or price >= 0.99:
                continue
            # Determine side and compute Kelly
            if prob > price:
                # Buy YES
                b = (1.0 - price) / price
                q = 1.0 - prob
                f = (b * prob - q) / b
                bet_price = price
                won = outcome == 1
            else:
                # Buy NO
                b = price / (1.0 - price)
                q = prob
                f = (b * (1.0 - prob) - q) / b
                bet_price = 1.0 - price
                won = outcome == 0
            f = max(f, 0.0) * kelly_fraction
            if f < 0.005:  # skip tiny bets
                continue
            stake = bankroll * f  # always size from flat bankroll
            if won:
                pnl += stake * (1.0 - bet_price) / bet_price
            else:
                pnl -= stake
            n_bets += 1

        results.append({
            "fold": fold,
            "brier": brier,
            "log_loss": ll,
            "accuracy": acc,
            "f1": f1,
            "avg_edge": avg_edge,
            "sim_pnl": pnl,
            "final_bankroll": bankroll,
            "n_bets": n_bets,
            "n_test": len(test_idx),
        })

        print(f"  Fold {fold}: Brier={brier:.4f} LogLoss={ll:.4f} "
              f"Acc={acc:.1%} F1={f1:.2f} Edge={avg_edge:+.4f} "
              f"PnL=€{pnl:+.2f} ({n_bets} bets, final €{bankroll:.2f})")

    return results


def export_xgboost_model(model, scaler, output_path: str, feature_names: list):
    """
    Export the trained XGBoost model as JSON for Rust inference.

    If the ensemble contains XGBoost, export it directly.
    Otherwise, export the full sklearn pipeline metadata.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Try to extract the XGBoost model from the ensemble
    xgb_model = _find_xgb_model(model)

    if xgb_model:
        # Export native XGBoost JSON (can be loaded in Rust)
        xgb_path = str(output)
        xgb_model.save_model(xgb_path)
        print(f"Exported XGBoost model to {xgb_path}")

        # Also export scaler parameters for Rust
        scaler_path = output.with_suffix(".scaler.json")
        scaler_data = {
            "center": scaler.center_.tolist(),
            "scale": scaler.scale_.tolist(),
            "feature_names": feature_names,
            "n_features": len(feature_names),
        }
        with open(scaler_path, "w") as f:
            json.dump(scaler_data, f, indent=2)
        print(f"Exported scaler to {scaler_path}")
    else:
        print("Warning: No XGBoost model found in ensemble, exporting metadata only")

    # Always export ensemble metadata
    meta_path = output.with_suffix(".meta.json")
    meta = {
        "exported_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "model_type": "stacking_ensemble",
        "has_xgboost": HAS_XGBOOST,
        "has_lightgbm": HAS_LIGHTGBM,
        "scaler_center": scaler.center_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported metadata to {meta_path}")


def _find_xgb_model(model):
    """Walk through calibration/stacking wrappers to find the XGBoost estimator."""
    # CalibratedClassifierCV -> StackingClassifier -> XGBClassifier
    if hasattr(model, "calibrated_classifiers_"):
        for cal_est in model.calibrated_classifiers_:
            found = _find_xgb_model(cal_est.estimator)
            if found:
                return found

    # StackingClassifier stores fitted estimators in estimators_ (list of lists)
    if hasattr(model, "estimators_") and hasattr(model, "named_estimators_"):
        # named_estimators_ is a Bunch with named access
        if hasattr(model.named_estimators_, "xgb"):
            xgb_est = model.named_estimators_.xgb
            if hasattr(xgb_est, "save_model"):
                return xgb_est

    # Direct XGBoost model
    if hasattr(model, "save_model") and hasattr(model, "get_booster"):
        return model

    return None


def _find_fitted_estimators(model):
    """Walk wrappers and yield (name, estimator) pairs with feature_importances_."""
    if hasattr(model, "calibrated_classifiers_"):
        for cal_est in model.calibrated_classifiers_:
            yield from _find_fitted_estimators(cal_est.estimator)
            return  # Only need first calibrated classifier

    if hasattr(model, "named_estimators_"):
        for name in model.named_estimators_:
            est = getattr(model.named_estimators_, name)
            if hasattr(est, "feature_importances_"):
                yield name, est


def print_feature_importance(model, feature_names):
    """Print feature importance from the ensemble."""
    print("\nFeature importance:")

    for name, est in _find_fitted_estimators(model):
        imp = est.feature_importances_
        pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
        print(f"\n  [{name}]:")
        for fname, val in pairs:
            bar = "#" * int(val * 100)
            print(f"    {fname:20s} {val:.4f} {bar}")


def main():
    parser = argparse.ArgumentParser(description="Train prediction model")
    parser.add_argument("--input", default="model/training_data.json")
    parser.add_argument("--output", default="model/xgb_model.json")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Skip probability calibration")
    args = parser.parse_args()

    # Load data
    df = load_data(args.input)
    if len(df) < 50:
        print(f"Error: Need at least 50 samples, got {len(df)}", file=sys.stderr)
        sys.exit(1)

    X = build_feature_matrix(df)
    y = df["label"].values
    prices = df["yes_price"].values

    print(f"\nFeature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Class balance: {y.mean():.1%} YES / {1-y.mean():.1%} NO")

    # Scale features (preserve DataFrame with column names)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    # Build ensemble
    print("\nBuilding ensemble model...")
    raw_model = build_ensemble()

    # Evaluate with time-series CV
    market_ids = df["market_id"].values if "market_id" in df.columns else None
    print("\nTime-series cross-validation:")
    results = evaluate_folds(raw_model, X_scaled, y, prices, market_ids=market_ids)

    avg_brier = np.mean([r["brier"] for r in results])
    avg_edge = np.mean([r["avg_edge"] for r in results])
    avg_pnl = np.mean([r["sim_pnl"] for r in results])
    total_bets = sum(r["n_bets"] for r in results)
    print(f"\nAvg Brier: {avg_brier:.4f} | Avg Edge: {avg_edge:+.4f} | "
          f"Avg PnL per fold: €{avg_pnl:+.2f} | Total bets: {total_bets}")

    # Train final model on all data with calibration
    print("\nTraining final model on all data...")
    if not args.no_calibration:
        # Isotonic calibration for well-calibrated probabilities
        n_samples = len(X_scaled)
        cal_method = "isotonic" if n_samples > 1000 else "sigmoid"
        cal_cv = min(3, max(2, n_samples // 200))
        print(f"  Calibration: {cal_method} with {cal_cv}-fold CV")

        model = CalibratedClassifierCV(
            estimator=raw_model,
            method=cal_method,
            cv=cal_cv,
            ensemble=True,
        )
    else:
        model = raw_model

    model.fit(X_scaled, y)

    # Final evaluation on last 20% (held out by time)
    split_idx = int(len(X_scaled) * 0.8)
    X_test_all = X_scaled.iloc[split_idx:]
    y_test_all = y[split_idx:]
    p_test_all = prices[split_idx:]

    if len(X_test_all) > 10:
        probs_all = model.predict_proba(X_test_all)[:, 1]
        print(f"\nHeld-out test ({len(X_test_all)} samples):")
        print(f"  Brier: {brier_score_loss(y_test_all, probs_all):.4f}")
        print(f"  Accuracy: {accuracy_score(y_test_all, (probs_all > 0.5).astype(int)):.1%}")

        # Report new-market-only subset for honest signal estimate
        if "market_id" in df.columns:
            train_ids = set(df["market_id"].iloc[:split_idx])
            test_ids = df["market_id"].iloc[split_idx:].values
            clean = np.array([mid not in train_ids for mid in test_ids])
            n_clean = clean.sum()
            if n_clean >= 10:
                probs_c = probs_all[clean]
                y_c = y_test_all[clean]
                p_c = p_test_all[clean]
                print(f"  New-market subset ({n_clean}/{len(X_test_all)} samples):")
                print(f"    Brier: {brier_score_loss(y_c, probs_c):.4f} "
                      f"(market baseline: {brier_score_loss(y_c, p_c):.4f})")
                print(f"    Accuracy: {accuracy_score(y_c, (probs_c > 0.5).astype(int)):.1%}")

        # Calibration curve
        try:
            frac, mean_pred = calibration_curve(y_test_all, probs_all, n_bins=5)
            print(f"  Calibration (predicted -> actual):")
            for pred, actual in zip(mean_pred, frac):
                print(f"    {pred:.2f} -> {actual:.2f}")
        except Exception:
            pass

    # Export
    print(f"\nExporting model...")
    export_xgboost_model(model, scaler, args.output, FEATURE_COLS)

    # Save full ensemble for the sidecar server
    ensemble_path = Path(args.output).parent / "ensemble.joblib"
    scaler_joblib_path = Path(args.output).parent / "scaler.joblib"
    joblib.dump(model, ensemble_path)
    joblib.dump(scaler, scaler_joblib_path)
    print(f"Saved full ensemble to {ensemble_path}")

    # Feature importance
    print_feature_importance(model, FEATURE_COLS)

    print("\nDone!")


if __name__ == "__main__":
    main()
