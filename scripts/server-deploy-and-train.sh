#!/usr/bin/env bash
# Eseguire SUL SERVER dopo push su github.com/valeriozanzarelli-debug/mind-runtime
# Oppure: CI workflow deploy.yml (rsync da GitHub Actions)
set -euo pipefail

REPO_URL="${MIND_RUNTIME_REPO:-https://github.com/valeriozanzarelli-debug/mind-runtime.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/mind-runtime}"
VENV="${INSTALL_DIR}/.venv"
LOG="/var/log/organism_train_integrated.log"
PIDFILE="/var/run/organism_train_integrated.pid"

echo "=== mind-runtime @ ${INSTALL_DIR} ==="

if [[ -d "${INSTALL_DIR}/.git" ]]; then
  echo "=== git pull ==="
  git -C "${INSTALL_DIR}" fetch origin main
  git -C "${INSTALL_DIR}" reset --hard origin/main
else
  echo "=== git clone (prima installazione) ==="
  BACKUP=""
  if [[ -d "${INSTALL_DIR}" ]]; then
    BACKUP="/tmp/mind-runtime-backup-$$"
    mkdir -p "$BACKUP"
    cp -a "${INSTALL_DIR}/data" "${INSTALL_DIR}/.venv" "$BACKUP/" 2>/dev/null || true
    rm -rf "${INSTALL_DIR}"
  fi
  git clone --depth 1 "$REPO_URL" "${INSTALL_DIR}"
  if [[ -n "$BACKUP" ]]; then
    cp -an "$BACKUP/data/." "${INSTALL_DIR}/data/" 2>/dev/null || true
    cp -a "$BACKUP/.venv" "${INSTALL_DIR}/.venv" 2>/dev/null || true
    rm -rf "$BACKUP"
  fi
fi

export ORGANISM_DNA_VARIANT="${ORGANISM_DNA_VARIANT:-genesis}"
python3 -m venv "${VENV}" 2>/dev/null || true
"${VENV}/bin/pip" install -q --upgrade pip pyyaml
if [[ -f "${INSTALL_DIR}/pyproject.toml" ]]; then
  "${VENV}/bin/pip" install -q -e "${INSTALL_DIR}[full]" 2>/dev/null \
    || "${VENV}/bin/pip" install -q -e "${INSTALL_DIR}" 2>/dev/null || true
fi

UNIT="${INSTALL_DIR}/deploy/systemd/organism-nursery.service"
if [[ -f "$UNIT" ]]; then
  sed -i "s/^Environment=ORGANISM_DNA_VARIANT=.*/Environment=ORGANISM_DNA_VARIANT=${ORGANISM_DNA_VARIANT}/" "$UNIT" || true
  cp "$UNIT" /etc/systemd/system/
  systemctl daemon-reload
  systemctl restart organism-nursery.service
  sleep 5
fi

chown www-data:www-data "${INSTALL_DIR}/data/baby_state.json" 2>/dev/null || true
chmod 664 "${INSTALL_DIR}/data/baby_state.json" 2>/dev/null || true

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Training già attivo pid=$(cat $PIDFILE)"
  exit 0
fi

TRAIN_CYCLES="${TRAIN_CYCLES:-10000}"
BENCHMARK_EVERY="${BENCHMARK_EVERY:-1000}"

nohup env \
  PYTHONPATH="${INSTALL_DIR}" \
  ORGANISM_DNA_VARIANT="${ORGANISM_DNA_VARIANT}" \
  TRAIN_CYCLES="${TRAIN_CYCLES}" \
  BENCHMARK_EVERY="${BENCHMARK_EVERY}" \
  BENCHMARK_LIMIT=100 \
  TARGET_COHERENT=0.90 \
  TRAIN_LOG="${LOG}" \
  "${VENV}/bin/python3" "${INSTALL_DIR}/scripts/train_integrated.py" \
  >> "${LOG}" 2>&1 &
echo $! > "$PIDFILE"
echo "Training avviato pid=$(cat $PIDFILE)"
echo "Monitor: tail -f ${LOG}"
