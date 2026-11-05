#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/smart-stock
mkdir -p logs
exec 9>logs/daily_replay.lock
flock -n 9 || exit 0
# Stable on same UTC day: retries do not advance another historical day.
execution_id="scheduled-$(date -u +%Y-%m-%d)"
exec .venv/bin/python -u scripts/daily_replay.py --config config/replay.json --execution-id "$execution_id"
