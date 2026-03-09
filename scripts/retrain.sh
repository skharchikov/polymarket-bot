#!/bin/sh
set -e

echo "$(date -Iseconds) Starting retrain cycle"

echo "Fetching resolved markets + own bets..."
python fetch_data.py --markets 1000 --output /model/training_data.json

echo "Training model..."
python train_model.py --input /model/training_data.json --output /model/xgb_model.json

echo "$(date -Iseconds) Retrain complete"
ls -lh /model/xgb_model.* /model/ensemble.joblib /model/scaler.joblib 2>/dev/null || true

# Tell the sidecar to reload the new model
if command -v curl >/dev/null 2>&1; then
    echo "Reloading model server..."
    curl -sf -X POST http://model-server:8000/reload && echo " OK" || echo " (sidecar not reachable)"
fi
