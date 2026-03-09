#!/usr/bin/env bash
# Install on Hetzner: copy to /opt/polymarket-bot/ and add crontab entry:
#   0 4 * * * /opt/polymarket-bot/retrain-cron.sh >> /opt/polymarket-bot/retrain.log 2>&1
#
# Retrains the XGBoost model daily at 4 AM UTC.
# The bot hot-reloads the model file on next scan cycle.

set -euo pipefail
cd /opt/polymarket-bot

echo "$(date -Iseconds) Starting scheduled retrain"
docker compose --profile train run --rm trainer
echo "$(date -Iseconds) Retrain finished"
