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

import json
import logging
import os
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Suppress sklearn feature-name warnings: base estimators fitted with feature
# names warn when called with a DataFrame slice; predictions are correct.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        # Uvicorn access logger passes a 5-tuple as args:
        #   (client_addr, method, full_path, http_version, status_code)
        if record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
            client, method, path, _version, status = record.args
            payload: dict = {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
                "level": "info",
                "logger": "access",
                "method": method,
                "path": path,
                "status": status,
                "client": client,
            }
        else:
            payload = {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
                "level": record.levelname.lower(),
                "logger": record.name,
                "msg": record.getMessage(),
            }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _HealthFilter(logging.Filter):
    """Drop /health access log records — they're sampled every ~5 s and add no value."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
            path = record.args[2]
            return path != "/health"
        return True


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_HealthFilter())

    # Replace the root handler so all loggers (ours + libraries) emit JSON.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]

    # Uvicorn manages its own loggers — redirect them to our handler.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False


_configure_logging()
log = logging.getLogger("sidecar")

# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
MODEL_PATH = Path(MODEL_DIR) / "ensemble.joblib"
SCALER_PATH = Path(MODEL_DIR) / "scaler.joblib"
ENCODING_PATH = Path(MODEL_DIR) / "category_encoding.json"
MAX_AGE_HOURS = int(os.environ.get("RETRAIN_MAX_AGE_HOURS", "24"))
RETRAIN_ON_SCHEDULE = os.environ.get("RETRAIN_ON_SCHEDULE", "true").lower() == "true"
RETRAIN_MARKETS = int(os.environ.get("RETRAIN_MARKETS", "3000"))
DATABASE_URL = os.environ.get("DATABASE_URL")
# Trigger warm-start retrain after this many new resolved bets accumulate
WARMSTART_TRIGGER_N = int(os.environ.get("WARMSTART_TRIGGER_N", "10"))

# Feature names — must stay in sync with MarketFeatures::NAMES in src/model/features.rs
# v3: removed log_liquidity, is_politics, is_sports (zero SHAP importance).
FEATURE_NAMES = [
    "yes_price", "momentum_1h", "momentum_24h", "volatility_24h", "rsi",
    "log_volume", "days_to_expiry",
    "is_crypto",
    "price_change_1d", "price_change_1w",
    "days_since_created", "created_to_expiry_span",
]

# Category columns subject to target encoding (binary → historical YES rate)
CATEGORY_COLS = ["is_crypto"]

app = FastAPI(title="Polymarket ML Sidecar")

# Global state
model = None
scaler = None
model_loaded_at = None
# Category encoding map: {col: {"0": rate, "1": rate}} — loaded from model artifact
category_encoding: dict[str, dict[str, float]] = {}


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
    global model, scaler, model_loaded_at, category_encoding
    if not MODEL_PATH.exists():
        log.warning("Model not found at %s, waiting for training...", MODEL_PATH)
        return False
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None

    # Detect feature schema mismatch between the loaded model and current FEATURE_NAMES.
    # This happens when a new model is deployed but the artifact on disk was trained on
    # a different feature set (e.g. old prod model with 13 features vs new code with 12).
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)
        if scaler_features != FEATURE_NAMES:
            log.error(
                "model.schema_mismatch: scaler has features %s but expected %s — triggering retrain",
                scaler_features,
                FEATURE_NAMES,
            )
            threading.Thread(target=_force_retrain, daemon=True).start()
            return False

    model_loaded_at = time.time()
    if ENCODING_PATH.exists():
        import json
        with open(ENCODING_PATH) as f:
            category_encoding = json.load(f)
        log.info("Loaded category encoding from %s", ENCODING_PATH)
    else:
        category_encoding = {}
        log.warning("No category_encoding.json found — category features will use raw binary values")
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
        log.info("retrain.fetch_start: markets=%d", RETRAIN_MARKETS)
        fetch_args = [
            sys.executable, "fetch_data.py",
            "--markets", str(RETRAIN_MARKETS),
            "--output", f"{MODEL_DIR}/training_data.json",
        ]
        if DATABASE_URL:
            fetch_args += ["--db", DATABASE_URL]
        subprocess.run(fetch_args, check=True, capture_output=True, text=True)

        log.info("retrain.train_start")
        subprocess.run(
            [sys.executable, "train_model.py",
             "--input", f"{MODEL_DIR}/training_data.json",
             "--output", f"{MODEL_DIR}/xgb_model.json"],
            check=True, capture_output=True, text=True,
        )

        load_model()
        _retrain.status = RetrainStatus.success
        log.info("retrain.complete")
    except subprocess.CalledProcessError as e:
        _retrain.status = RetrainStatus.failed
        _retrain.error = e.stderr[-500:] if e.stderr else str(e)
        log.error("retrain.failed: %s", _retrain.error)
    except Exception as e:
        _retrain.status = RetrainStatus.failed
        _retrain.error = str(e)
        log.error("retrain.failed: %s", e)
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


def _force_retrain():
    """Trigger an unconditional retrain, regardless of model age."""
    if not _retrain.lock.acquire(blocking=False):
        log.info("retrain.skip: already in progress")
        return
    try:
        _run_retrain()
    finally:
        _retrain.lock.release()


def _retrain_if_stale():
    """Retrain only if the model is missing or older than MAX_AGE_HOURS."""
    age = model_age_hours()
    if age is not None and age < MAX_AGE_HOURS:
        log.info("retrain.skip: model is %.1fh old (max %dh)", age, MAX_AGE_HOURS)
        return
    if age is not None:
        log.info("retrain.stale: model is %.1fh old, retraining", age)
    else:
        log.info("retrain.no_model: training from scratch")

    if not _retrain.lock.acquire(blocking=False):
        log.info("retrain.skip: already in progress")
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
            log.error("retrain.schedule_error: %s", e)


@app.on_event("startup")
def startup():
    if not MODEL_PATH.exists():
        _retrain_if_stale()
    else:
        load_model()

    if RETRAIN_ON_SCHEDULE:
        t = threading.Thread(target=_schedule_loop, daemon=True)
        t.start()
        log.info("retrain.scheduled: checking every 1h, max age %dh", MAX_AGE_HOURS)


class FeatureMap(BaseModel):
    """Named feature schema — must stay in sync with MarketFeatures in src/model/features.rs."""
    yes_price: float
    momentum_1h: float
    momentum_24h: float
    volatility_24h: float
    rsi: float
    log_volume: float
    days_to_expiry: float
    is_crypto: float
    price_change_1d: float
    price_change_1w: float
    days_since_created: float
    created_to_expiry_span: float

    def to_row(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in FEATURE_NAMES}


def _apply_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Apply target encoding to category columns if encoding map is loaded."""
    if not category_encoding:
        return df
    df = df.copy()
    for col in CATEGORY_COLS:
        if col in category_encoding and col in df.columns:
            enc = category_encoding[col]
            df[col] = df[col].apply(lambda v: enc.get(str(int(v)), enc.get("0", v)))
    return df


class PredictRequest(BaseModel):
    features: FeatureMap
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

    features = pd.DataFrame([req.features.model_dump()], dtype=np.float64)[FEATURE_NAMES]
    features = _apply_encoding(features)
    if scaler is not None:
        features = pd.DataFrame(scaler.transform(features), columns=FEATURE_NAMES)

    prob = float(model.predict_proba(features)[0, 1])
    confidence = _estimate_confidence(features)

    log.info(
        "predict: price=%.1f%% prob=%.1f%% conf=%.0f%% edge=%+.1f%%",
        req.market_price * 100,
        prob * 100,
        confidence * 100,
        (prob - req.market_price) * 100,
    )

    return PredictResponse(prob=prob, confidence=confidence)


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: PredictBatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.items:
        return BatchResponse(predictions=[])

    features = pd.DataFrame([item.features.model_dump() for item in req.items], dtype=np.float64)[FEATURE_NAMES]
    features = _apply_encoding(features)
    if scaler is not None:
        features = pd.DataFrame(scaler.transform(features), columns=FEATURE_NAMES)

    probs = model.predict_proba(features)[:, 1]
    confidences = [_estimate_confidence(features.iloc[i:i+1]) for i in range(len(features))]
    market_prices = [item.market_price for item in req.items]

    log.info(
        "predict_batch: n=%d avg_conf=%.0f%% avg_edge=%+.1f%%",
        len(probs),
        np.mean(confidences) * 100,
        np.mean(probs - np.array(market_prices)) * 100,
    )

    return BatchResponse(
        predictions=[
            PredictResponse(prob=float(p), confidence=c)
            for p, c in zip(probs, confidences)
        ]
    )


def _estimate_confidence(features: pd.DataFrame) -> float:
    """Estimate confidence from base estimator disagreement.

    Accepts a single-row DataFrame (with column names) so base estimators
    that were fitted with feature names don't emit validation warnings.
    """
    try:
        base_preds = []

        estimator = model
        if hasattr(estimator, "calibrated_classifiers_"):
            estimator = estimator.calibrated_classifiers_[0].estimator

        if hasattr(estimator, "estimators_"):
            for est_list in estimator.estimators_:
                if isinstance(est_list, list):
                    for est in est_list:
                        base_preds.append(est.predict_proba(features)[0, 1])
                else:
                    base_preds.append(est_list.predict_proba(features)[0, 1])

        if len(base_preds) >= 2:
            spread = max(base_preds) - min(base_preds)
            # spread 0 → conf 0.75, spread 0.3 → conf ~0.36, spread 0.5+ → conf 0.25
            confidence = 0.75 / (1.0 + spread * 4.0)
            return max(0.25, min(0.75, confidence))
    except Exception:
        pass

    return 0.50


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
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)
