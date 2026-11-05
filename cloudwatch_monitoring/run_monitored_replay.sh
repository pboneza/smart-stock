#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/smart-stock
mkdir -p logs
exec 9>logs/daily_replay.lock
flock -n 9 || exit 0
execution_id="scheduled-$(date -u +%Y-%m-%d)"
exec .venv/bin/python -u scripts/cloudwatch_replay.py --config config/replay.json --monitor-config config/monitoring.json --mode run --execution-id "$execution_id"
