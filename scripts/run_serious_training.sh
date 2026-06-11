#!/usr/bin/env bash
# Training serio in background — ferma nursery per evitare corruzione stato.
set -eu
cd /opt/mind-runtime
LOG=/tmp/organism_train_serious.log
export MEGA_CURRICULUM_LIMIT="${MEGA_CURRICULUM_LIMIT:-1000}"
export MEGA_CURRICULUM_PAUSE="${MEGA_CURRICULUM_PAUSE:-0.22}"
export VOCAB_SENTENCES_LIMIT="${VOCAB_SENTENCES_LIMIT:-8000}"
export TRAIN_PLASTICITY="${TRAIN_PLASTICITY:-0.025}"

echo "=== stop nursery (training esclusivo stato) ===" | tee "$LOG"
systemctl stop organism-nursery.service || true
sleep 2

echo "=== train_serious start $(date -Iseconds) ===" | tee -a "$LOG"
if ! .venv/bin/python scripts/train_serious.py 2>&1 | tee -a "$LOG"; then
  echo "=== TRAINING FAILED — riavvio nursery comunque ===" | tee -a "$LOG"
fi

echo "=== start nursery $(date -Iseconds) ===" | tee -a "$LOG"
systemctl start organism-nursery.service
sleep 5
systemctl is-active organism-nursery.service | tee -a "$LOG"
echo "=== done — log: $LOG ==="
