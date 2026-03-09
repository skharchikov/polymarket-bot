#!/bin/sh
set -e

MODEL_FILE="/model/ensemble.joblib"
MAX_AGE_HOURS="${RETRAIN_MAX_AGE_HOURS:-72}"

# Skip training if the model is fresh enough
if [ -f "$MODEL_FILE" ]; then
    # stat -c works on Linux (Debian), stat -f on macOS
    mod_time=$(stat -c %Y "$MODEL_FILE" 2>/dev/null || stat -f %m "$MODEL_FILE" 2>/dev/null || echo 0)
    now=$(date +%s)
    age_secs=$(( now - mod_time ))
    age_hours=$(( age_secs / 3600 ))
    echo "Existing model is ${age_hours}h old (max ${MAX_AGE_HOURS}h)"
    if [ "$age_hours" -lt "$MAX_AGE_HOURS" ]; then
        echo "Model is fresh - skipping retrain"
        exit 0
    fi
fi

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
