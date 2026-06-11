#!/usr/bin/env bash
# Installa ORGANISM su server ink (es. ssh ink)
# Uso: sudo bash deploy/install-on-ink-server.sh
set -euo pipefail

INSTALL_DIR=/opt/mind-runtime
REPO_SRC="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${INSTALL_DIR}/.venv"

echo "==> Install mind-runtime in ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude '.git' --exclude '.pytest_cache' --exclude '__pycache__' --exclude '*.egg-info' --exclude '.venv' \
    "${REPO_SRC}/" "${INSTALL_DIR}/"
else
  echo "    (rsync assente — copia da ${REPO_SRC})"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 non trovato" >&2
  exit 1
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip
fi

python3 -m venv "${VENV}"
"${VENV}/bin/pip" install -q --upgrade pip
"${VENV}/bin/pip" install -q -e "${INSTALL_DIR}[full]"

echo "==> data dir (persistenza baby)"
mkdir -p "${INSTALL_DIR}/data"
chown -R www-data:www-data "${INSTALL_DIR}/data"

echo "==> systemd organism-nursery.service"
cp "${INSTALL_DIR}/deploy/systemd/organism-nursery.service" /etc/systemd/system/
chown -R www-data:www-data "${INSTALL_DIR}"
systemctl daemon-reload
systemctl enable organism-nursery.service
systemctl restart organism-nursery.service
systemctl --no-pager status organism-nursery.service || true

echo ""
echo "==> Caddy: aggiungi il blocco da ${INSTALL_DIR}/deploy/caddy/organism.caddy"
echo "    poi: sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy"
echo ""
echo "==> URL: https://inkconscius.eu/organism/"
echo "    (microfono/camera richiedono HTTPS — Chrome)"
