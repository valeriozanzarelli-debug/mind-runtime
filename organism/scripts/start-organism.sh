#!/usr/bin/env bash
# Riaccende organism-nursery (profilo baby di default).
set -euo pipefail

HOST="${DEPLOY_HOST:-46.225.222.101}"
VARIANT="${ORGANISM_DNA_VARIANT:-baby}"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)

if [[ -n "${SSH_PRIVATE_KEY:-}" ]]; then
  KEY="${KEY_FILE:-/tmp/mind_start_key.$$}"
  if [[ -f "$(dirname "$0")/../../ci/write_deploy_key.sh" ]]; then
    bash "$(dirname "$0")/../../ci/write_deploy_key.sh" "$KEY"
  else
    printf '%s\n' "$SSH_PRIVATE_KEY" >"$KEY"
    chmod 600 "$KEY"
  fi
  SSH_OPTS+=(-i "$KEY" -o IdentitiesOnly=yes)
  trap 'rm -f "$KEY"' EXIT
fi

echo "==> Start organism-nursery @ ${HOST} (variant=${VARIANT})"
ssh "${SSH_OPTS[@]}" "root@${HOST}" \
  "sed -i 's/^Environment=ORGANISM_DNA_VARIANT=.*/Environment=ORGANISM_DNA_VARIANT=${VARIANT}/' /etc/systemd/system/organism-nursery.service 2>/dev/null || true; systemctl daemon-reload; systemctl enable organism-nursery.service; systemctl start organism-nursery.service; sleep 2; systemctl is-active organism-nursery.service"

echo "==> https://inkconscius.eu/organism/"
