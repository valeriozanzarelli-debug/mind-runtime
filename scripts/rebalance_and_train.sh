#!/usr/bin/env bash
# Squash lessico + training semantico in locale (no gateway timeout).
set -euo pipefail
cd /opt/mind-runtime
BASE="${1:-http://127.0.0.1:8765/organism}"
VENV=".venv/bin/python"

echo "=== squash lessico ==="
curl -sf -X POST "${BASE}/api/baby/sense" -H 'Content-Type: application/json' -d '{"text":"__warmup__"}' >/dev/null || true

echo "=== train emergent language (localhost) ==="
pkill -f 'train_emergent_language.py' 2>/dev/null || true
nohup "$VENV" scripts/train_emergent_language.py "$BASE" > /tmp/organism_train_language.log 2>&1 &
echo "PID $! — tail -f /tmp/organism_train_language.log"
