#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/smart-stock
exec .venv/bin/python -u scripts/cloudwatch_replay.py --config config/replay.json --monitor-config config/monitoring.json --mode watchdog
