#!/bin/sh
set -e

echo "$(date -Iseconds) Starting retrain cycle"

echo "Fetching resolved markets..."
python fetch_data.py --markets 1000 --output /model/training_data.json

echo "Training model..."
python train_model.py --input /model/training_data.json --output /model/xgb_model.json

echo "$(date -Iseconds) Retrain complete"
ls -lh /model/xgb_model.*
