#!/usr/bin/env python3
"""
ML model sidecar — serves the full stacking ensemble via HTTP.

The Rust bot sends feature vectors, this returns calibrated probabilities
from the full ensemble (XGBoost + LightGBM + HistGBM + ExtraTrees + RF + meta-learner).

Endpoints:
    POST /predict        — single prediction
    POST /predict_batch  — batch predictions
    POST /reload         — reload model from disk
    POST /retrain        — trigger retrain (force=true to skip staleness check)
    GET  /retrain/status — check retrain status
    GET  /health         — liveness check
"""

import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Structured JSON logging — output is parseable by log aggregators (Loki, Datadog, etc.)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
log = structlog.get_logger("sidecar")

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
MODEL_PATH = Path(MODEL_DIR) / "ensemble.joblib"
SCALER_PATH = Path(MODEL_DIR) / "scaler.joblib"
MAX_AGE_HOURS = int(os.environ.get("RETRAIN_MAX_AGE_HOURS", "24"))
RETRAIN_ON_SCHEDULE = os.environ.get("RETRAIN_ON_SCHEDULE", "true").lower() == "true"
RETRAIN_MARKETS = int(os.environ.get("RETRAIN_MARKETS", "3000"))
DATABASE_URL = os.environ.get("DATABASE_URL")
# Trigger warm-start retrain after this many new resolved bets accumulate
WARMSTART_TRIGGER_N = int(os.environ.get("WARMSTART_TRIGGER_N", "10"))

FEATURE_NAMES = [
    "yes_price", "momentum_1h", "momentum_24h", "volatility_24h", "rsi",
    "log_volume", "log_liquidity", "days_to_expiry",
    "is_crypto", "is_politics", "is_sports",
    "price_change_1d", "price_change_1w",
]

app = FastAPI(title="Polymarket ML Sidecar")

# Global state
model = None
scaler = None
model_loaded_at = None


class RetrainStatus(str, Enum):
    idle = "idle"
    running = "running"
    success = "success"
    failed = "failed"


class _RetrainState:
    def __init__(self):
        self.status: RetrainStatus = RetrainStatus.idle
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.error: str | None = None
        self.lock = threading.Lock()


_retrain = _RetrainState()


def load_model():
    global model, scaler, model_loaded_at
    if not MODEL_PATH.exists():
        log.warning("Model not found at %s, waiting for training...", MODEL_PATH)
        return False
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    model_loaded_at = time.time()
    log.info("Loaded ensemble from %s", MODEL_PATH)
    return True


def model_age_hours() -> float | None:
    if not MODEL_PATH.exists():
        return None
    return (time.time() - MODEL_PATH.stat().st_mtime) / 3600


def _run_retrain():
    """Full cold retrain: fetch N markets + train from scratch, then reload."""
    _retrain.status = RetrainStatus.running
    _retrain.started_at = datetime.now(timezone.utc).isoformat()
    _retrain.finished_at = None
    _retrain.error = None

    try:
        log.info("retrain.start", kind="cold", markets=RETRAIN_MARKETS)
        fetch_args = [
            sys.executable, "fetch_data.py",
            "--markets", str(RETRAIN_MARKETS),
            "--output", f"{MODEL_DIR}/training_data.json",
        ]
        if DATABASE_URL:
            fetch_args += ["--db", DATABASE_URL]
        subprocess.run(fetch_args, check=True, capture_output=True, text=True)

        log.info("retrain.train", kind="cold")
        subprocess.run(
            [sys.executable, "train_model.py",
             "--input", f"{MODEL_DIR}/training_data.json",
             "--output", f"{MODEL_DIR}/xgb_model.json"],
            check=True, capture_output=True, text=True,
        )

        load_model()
        _retrain.status = RetrainStatus.success
        log.info("retrain.complete", kind="cold")
    except subprocess.CalledProcessError as e:
        _retrain.status = RetrainStatus.failed
        _retrain.error = e.stderr[-500:] if e.stderr else str(e)
        log.error("retrain.failed", kind="cold", error=_retrain.error)
    except Exception as e:
        _retrain.status = RetrainStatus.failed
        _retrain.error = str(e)
        log.error("retrain.failed", kind="cold", error=str(e))
    finally:
        _retrain.finished_at = datetime.now(timezone.utc).isoformat()


def _run_warmstart():
    """Warm-start retrain: add XGBoost trees on top of existing model using stored bet_features.

    No API calls — reads exact feature vectors from DB. Takes seconds.
    Triggered automatically every WARMSTART_TRIGGER_N resolved bets.
    """
    if not DATABASE_URL:
        log.warning("warmstart.skip", reason="DATABASE_URL not set")
        return

    try:
        import psycopg2
    except ImportError:
        log.warning("warmstart.skip", reason="psycopg2 not installed")
        return

    try:
        import xgboost as xgb
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.preprocessing import RobustScaler
    except ImportError as e:
        log.warning("warmstart.skip", reason=str(e))
        return

    log.info("warmstart.start", trigger_n=WARMSTART_TRIGGER_N)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT bf.features, b.won, b.side
            FROM bet_features bf
            JOIN bets b ON b.id = bf.bet_id
            WHERE b.resolved = TRUE AND b.won IS NOT NULL
            ORDER BY b.resolved_at DESC
        """)
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        log.error("warmstart.db_error", error=str(e))
        return

    if len(rows) < 10:
        log.info("warmstart.skip", reason="not enough resolved bets", n=len(rows))
        return

    # Build feature matrix from stored JSONB
    records = []
    labels = []
    for features_json, won, side in rows:
        if features_json is None or won is None:
            continue
        row = {k: float(v) if v is not None else 0.0 for k, v in features_json.items()}
        outcome_yes = won if side == "Yes" else not won
        records.append(row)
        labels.append(int(outcome_yes))

    if len(records) < 10:
        log.info("warmstart.skip", reason="not enough clean records", n=len(records))
        return

    df = pd.DataFrame(records)
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    X = df[FEATURE_NAMES].astype(np.float64).fillna(0.0)
    y = np.array(labels)

    # Scale using existing scaler
    if scaler is not None:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_NAMES)
    else:
        X_scaled = X

    # Warm-start XGBoost: continue from existing model, add 50 trees
    xgb_path = Path(MODEL_DIR) / "xgb_model.json"
    if not xgb_path.exists():
        log.warning("warmstart.skip", reason="no existing xgb_model.json")
        return

    try:
        booster = xgb.XGBClassifier(
            n_estimators=50,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        booster.fit(X_scaled, y, xgb_model=str(xgb_path))
        booster.save_model(str(xgb_path))
        log.info("warmstart.xgb_updated", n_samples=len(records), trees_added=50)
    except Exception as e:
        log.error("warmstart.xgb_failed", error=str(e))
        return

    # Hot-reload so next predictions use the updated model
    load_model()
    log.info("warmstart.complete", n_samples=len(records))


def _retrain_if_stale():
    """Retrain only if the model is missing or older than MAX_AGE_HOURS."""
    age = model_age_hours()
    if age is not None and age < MAX_AGE_HOURS:
        log.info("Model is %.1fh old (max %dh) — fresh, skipping", age, MAX_AGE_HOURS)
        return
    if age is not None:
        log.info("Model is %.1fh old (max %dh) — stale, retraining", age, MAX_AGE_HOURS)
    else:
        log.info("No model found — training")

    if not _retrain.lock.acquire(blocking=False):
        log.info("Retrain already in progress, skipping scheduled run")
        return
    try:
        _run_retrain()
    finally:
        _retrain.lock.release()


def _schedule_loop():
    """Background thread: check staleness every hour, retrain if needed."""
    while True:
        time.sleep(3600)
        try:
            _retrain_if_stale()
        except Exception as e:
            log.error("Scheduled retrain check failed: %s", e)


@app.on_event("startup")
def startup():
    # Train on first boot if no model exists
    if not MODEL_PATH.exists():
        _retrain_if_stale()
    else:
        load_model()

    if RETRAIN_ON_SCHEDULE:
        t = threading.Thread(target=_schedule_loop, daemon=True)
        t.start()
        log.info("Scheduled retrain: checking every 1h, max model age %dh", MAX_AGE_HOURS)


class PredictRequest(BaseModel):
    features: list[float]
    market_price: float = 0.5


class PredictBatchRequest(BaseModel):
    items: list[PredictRequest]


class PredictResponse(BaseModel):
    prob: float
    confidence: float


class BatchResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_age_secs: float | None = None


@app.get("/health", response_model=HealthResponse)
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — run trainer first",
        )
    age = time.time() - model_loaded_at if model_loaded_at else None
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_age_secs=age,
    )


@app.post("/reload")
def reload():
    ok = load_model()
    if not ok:
        raise HTTPException(status_code=503, detail="Model file not found")
    return {"status": "reloaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = pd.DataFrame([req.features], columns=FEATURE_NAMES, dtype=np.float64)
    if scaler is not None:
        features = pd.DataFrame(scaler.transform(features), columns=FEATURE_NAMES)

    prob = float(model.predict_proba(features)[0, 1])
    confidence = _estimate_confidence(features.values)

    log.info(
        "predict",
        price_pct=round(req.market_price * 100, 1),
        prob_pct=round(prob * 100, 1),
        conf_pct=round(confidence * 100, 0),
        edge_pct=round((prob - req.market_price) * 100, 1),
    )

    return PredictResponse(prob=prob, confidence=confidence)


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: PredictBatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.items:
        return BatchResponse(predictions=[])

    features = pd.DataFrame([item.features for item in req.items], columns=FEATURE_NAMES, dtype=np.float64)
    if scaler is not None:
        features = pd.DataFrame(scaler.transform(features), columns=FEATURE_NAMES)

    probs = model.predict_proba(features)[:, 1]
    confidences = [_estimate_confidence(features.values[i : i + 1]) for i in range(len(features))]
    market_prices = [item.market_price for item in req.items]

    log.info(
        "predict_batch",
        n=len(probs),
        avg_conf_pct=round(float(np.mean(confidences)) * 100, 0),
        avg_edge_pct=round(float(np.mean(probs - np.array(market_prices))) * 100, 1),
    )

    return BatchResponse(
        predictions=[
            PredictResponse(prob=float(p), confidence=c)
            for p, c in zip(probs, confidences)
        ]
    )


def _estimate_confidence(features: np.ndarray) -> float:
    """Estimate confidence from base estimator disagreement.

    For a stacking ensemble: get predictions from each base model,
    measure their spread. Low spread = high agreement = high confidence.
    """
    try:
        # Try to get base estimator predictions
        base_preds = []

        # CalibratedClassifierCV wraps the stacker
        estimator = model
        if hasattr(estimator, "calibrated_classifiers_"):
            estimator = estimator.calibrated_classifiers_[0].estimator

        if hasattr(estimator, "estimators_"):
            # StackingClassifier — get predictions from each base model
            for est_list in estimator.estimators_:
                if isinstance(est_list, list):
                    for est in est_list:
                        pred = est.predict_proba(features)[0, 1]
                        base_preds.append(pred)
                else:
                    pred = est_list.predict_proba(features)[0, 1]
                    base_preds.append(pred)

        if len(base_preds) >= 2:
            spread = max(base_preds) - min(base_preds)
            # spread 0 → conf 0.75, spread 0.3 → conf ~0.36, spread 0.5+ → conf 0.25
            confidence = 0.75 / (1.0 + spread * 4.0)
            return max(0.25, min(0.75, confidence))
    except Exception:
        pass

    return 0.50  # Default moderate confidence


class RetrainRequest(BaseModel):
    force: bool = False


class RetrainStatusResponse(BaseModel):
    status: RetrainStatus
    model_age_hours: float | None = None
    max_age_hours: int = MAX_AGE_HOURS
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


@app.post("/retrain")
def retrain(req: RetrainRequest = RetrainRequest()):
    if _retrain.status == RetrainStatus.running:
        raise HTTPException(status_code=409, detail="Retrain already in progress")

    age = model_age_hours()
    if not req.force and age is not None and age < MAX_AGE_HOURS:
        return {
            "status": "skipped",
            "reason": f"Model is {age:.1f}h old (max {MAX_AGE_HOURS}h). Use force=true to override.",
        }

    if not _retrain.lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Retrain already in progress")

    def run():
        try:
            _run_retrain()
        finally:
            _retrain.lock.release()

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started"}


@app.post("/retrain/warmstart")
def retrain_warmstart():
    """Warm-start retrain on stored bet_features — no API calls, takes seconds."""
    if _retrain.status == RetrainStatus.running:
        raise HTTPException(status_code=409, detail="Retrain already in progress")
    threading.Thread(target=_run_warmstart, daemon=True).start()
    return {"status": "started", "kind": "warmstart"}


@app.get("/retrain/status", response_model=RetrainStatusResponse)
def retrain_status():
    return RetrainStatusResponse(
        status=_retrain.status,
        model_age_hours=model_age_hours(),
        started_at=_retrain.started_at,
        finished_at=_retrain.finished_at,
        error=_retrain.error,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MODEL_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
