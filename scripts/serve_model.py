#!/usr/bin/env python3
"""
ML model sidecar — serves the full stacking ensemble via HTTP.

The Rust bot sends feature vectors, this returns calibrated probabilities
from the full ensemble (XGBoost + LightGBM + HistGBM + ExtraTrees + RF + meta-learner).

Endpoints:
    POST /predict        — single prediction
    POST /predict_batch  — batch predictions
    GET  /health         — liveness check
"""

import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sidecar")

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
MODEL_PATH = Path(MODEL_DIR) / "ensemble.joblib"
SCALER_PATH = Path(MODEL_DIR) / "scaler.joblib"

FEATURE_NAMES = [
    "yes_price", "momentum_1h", "momentum_24h", "volatility_24h", "rsi",
    "log_volume", "log_liquidity", "days_to_expiry",
    "is_crypto", "is_politics", "is_sports",
    "news_count", "best_news_score", "avg_news_age_hours",
    "order_imbalance", "spread", "price_change_1d", "price_change_1w",
]

app = FastAPI(title="Polymarket ML Sidecar")

# Global state
model = None
scaler = None
model_loaded_at = None


def load_model():
    global model, scaler, model_loaded_at
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}, waiting for training...")
        return False
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    model_loaded_at = time.time()
    print(f"Loaded ensemble from {MODEL_PATH}")
    return True


@app.on_event("startup")
def startup():
    load_model()


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
        "predict  price=%.1f%% → prob=%.1f%% conf=%.0f%% edge=%+.1f%%",
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

    features = pd.DataFrame([item.features for item in req.items], columns=FEATURE_NAMES, dtype=np.float64)
    if scaler is not None:
        features = pd.DataFrame(scaler.transform(features), columns=FEATURE_NAMES)

    probs = model.predict_proba(features)[:, 1]
    confidences = [_estimate_confidence(features.values[i : i + 1]) for i in range(len(features))]
    market_prices = [item.market_price for item in req.items]

    for price, prob, conf in zip(market_prices, probs, confidences):
        log.info(
            "batch    price=%.1f%% → prob=%.1f%% conf=%.0f%% edge=%+.1f%%",
            price * 100,
            float(prob) * 100,
            conf * 100,
            (float(prob) - price) * 100,
        )

    log.info("batch complete: %d predictions, avg_conf=%.0f%%", len(probs), np.mean(confidences) * 100)

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


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MODEL_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
