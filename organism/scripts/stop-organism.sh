#!/usr/bin/env bash
# Spegne organism-nursery sul server (non parte al reboot).
set -euo pipefail

HOST="${DEPLOY_HOST:-46.225.222.101}"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)

if [[ -n "${SSH_PRIVATE_KEY:-}" ]]; then
  KEY="${KEY_FILE:-/tmp/mind_stop_key.$$}"
  if [[ -f "$(dirname "$0")/../../ci/write_deploy_key.sh" ]]; then
    bash "$(dirname "$0")/../../ci/write_deploy_key.sh" "$KEY"
  else
    printf '%s\n' "$SSH_PRIVATE_KEY" >"$KEY"
    chmod 600 "$KEY"
  fi
  SSH_OPTS+=(-i "$KEY" -o IdentitiesOnly=yes)
  trap 'rm -f "$KEY"' EXIT
fi

echo "==> Stop organism-nursery @ ${HOST}"
ssh "${SSH_OPTS[@]}" "root@${HOST}" \
  'systemctl stop organism-nursery.service; systemctl disable organism-nursery.service; systemctl is-active organism-nursery.service || echo spento'
