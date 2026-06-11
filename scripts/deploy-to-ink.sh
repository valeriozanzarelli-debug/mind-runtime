#!/usr/bin/env bash
# Deploy mind-runtime su server ink (default: profilo baby, leggero in RAM).
# Uso: bash scripts/deploy-to-ink.sh
# Richiede: ssh root@46.225.222.101 (chiave cursor-cloud-agent)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${DEPLOY_HOST:-46.225.222.101}"
USER="${DEPLOY_USER:-root}"
TARGET="${SSH_TARGET:-${USER}@${HOST}}"
VARIANT="${ORGANISM_DNA_VARIANT:-adolescent}"
PUBLIC_URL="${ORGANISM_PUBLIC_URL:-https://inkconscius.eu/organism}"

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [[ -n "${SSH_PRIVATE_KEY:-}" ]]; then
  KEY_FILE="${KEY_FILE:-/tmp/mind_runtime_deploy_key.$$}"
  if [[ -f "${ROOT}/../ci/write_deploy_key.sh" ]]; then
    bash "${ROOT}/../ci/write_deploy_key.sh" "$KEY_FILE"
  else
    printf '%s\n' "$SSH_PRIVATE_KEY" >"$KEY_FILE"
    chmod 600 "$KEY_FILE"
  fi
  SSH_OPTS+=(-i "$KEY_FILE" -o IdentitiesOnly=yes)
  trap 'rm -f "$KEY_FILE"' EXIT
fi

echo "=== Deploy mind-runtime → ${TARGET} (variant=${VARIANT}) ==="

if command -v rsync >/dev/null 2>&1; then
  rsync -az --delete \
    --exclude '.git' --exclude '.pytest_cache' --exclude '__pycache__' \
    --exclude '*.egg-info' --exclude '.venv' --exclude 'data/baby_state.json' \
    --exclude 'data/*.organism.json' \
    "${ROOT}/" "${TARGET}:/opt/mind-runtime/"
else
  tar -C "${ROOT}" -czf - \
    --exclude='.git' --exclude='.pytest_cache' --exclude='__pycache__' \
    --exclude='*.egg-info' --exclude='.venv' --exclude='data' . \
    | ssh "${SSH_OPTS[@]}" "$TARGET" 'mkdir -p /opt/mind-runtime && tar -xzf - -C /opt/mind-runtime'
fi

ssh "${SSH_OPTS[@]}" "$TARGET" "ORGANISM_DNA_VARIANT=${VARIANT} bash -s" <<'REMOTE'
set -euo pipefail
INSTALL_DIR=/opt/mind-runtime
VENV="${INSTALL_DIR}/.venv"
VARIANT="${ORGANISM_DNA_VARIANT:-baby}"

echo "==> venv + pip"
if ! python3 -m venv --help >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip
fi
python3 -m venv "${VENV}"
"${VENV}/bin/pip" install -q --upgrade pip
"${VENV}/bin/pip" install -q -e "${INSTALL_DIR}[full]"

mkdir -p "${INSTALL_DIR}/data"
chown -R www-data:www-data "${INSTALL_DIR}/data" "${INSTALL_DIR}"

echo "==> systemd (ORGANISM_DNA_VARIANT=${VARIANT})"
UNIT="${INSTALL_DIR}/deploy/systemd/organism-nursery.service"
sed -i "s/^Environment=ORGANISM_DNA_VARIANT=.*/Environment=ORGANISM_DNA_VARIANT=${VARIANT}/" "$UNIT" 2>/dev/null || true
case "${VARIANT}" in
  adolescent)
    sed -i 's/^Environment=ORGANISM_MAX_NEURONS=.*/Environment=ORGANISM_MAX_NEURONS=35000/' "$UNIT" || true
    sed -i 's/^Environment=ORGANISM_GROWTH_BATCH=.*/Environment=ORGANISM_GROWTH_BATCH=120/' "$UNIT" || true
    sed -i 's/^Environment=ORGANISM_GROWTH_EVERY=.*/Environment=ORGANISM_GROWTH_EVERY=20/' "$UNIT" || true
    ;;
  baby)
    sed -i 's/^Environment=ORGANISM_MAX_NEURONS=.*/Environment=ORGANISM_MAX_NEURONS=20000/' "$UNIT" || true
    ;;
esac
if [[ "${SKIP_REBIRTH:-1}" == "0" ]]; then
  sed -i 's/^Environment=ORGANISM_ALLOW_REBIRTH=.*/Environment=ORGANISM_ALLOW_REBIRTH=1/' "$UNIT" || true
else
  sed -i 's/^Environment=ORGANISM_ALLOW_REBIRTH=.*/Environment=ORGANISM_ALLOW_REBIRTH=0/' "$UNIT" || true
fi
cp "$UNIT" /etc/systemd/system/
systemctl daemon-reload
systemctl restart organism-nursery.service
echo "==> attesa avvio servizio…"
sleep 5
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8765/organism/api/baby/state >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
systemctl --no-pager status organism-nursery.service || true
REMOTE

if [[ "${SKIP_REBIRTH:-1}" == "1" ]]; then
  echo "==> skip rebirth (memoria preservata, SKIP_REBIRTH=1) ==="
else
  echo "==> rebirth (conserva dialoghi) ==="
  sleep 3
  curl -sf -X POST "${PUBLIC_URL}/api/baby/rebirth" \
    -H 'Content-Type: application/json' \
    -d '{"keep_dialogue": true}' | python3 -m json.tool || {
    echo "WARN: rebirth via API fallito"
  }
fi

echo "==> stato finale ==="
curl -sf "${PUBLIC_URL}/api/baby/state" | python3 -c "
import json,sys
d=json.load(sys.stdin)
s=d.get('stats',{})
print(f'neuroni={s.get(\"neurons\")} sinapsi={s.get(\"synapses\")} species={s.get(\"species\")}')
print(f'dialoghi={len(d.get(\"dialogue_pairs\",[]))} parole={d.get(\"words_known\")}')
"

echo "=== Deploy completato ==="
