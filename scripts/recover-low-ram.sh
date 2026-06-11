#!/usr/bin/env bash
# Ripristina organism-nursery su profilo baby dopo OOM da giga/mega.
# Uso sul server: sudo bash scripts/recover-low-ram.sh
set -euo pipefail

UNIT=/etc/systemd/system/organism-nursery.service
INSTALL=/opt/mind-runtime

echo "==> RAM attuale"
free -h

if [[ -f "${INSTALL}/deploy/systemd/organism-nursery.service" ]]; then
  cp "${INSTALL}/deploy/systemd/organism-nursery.service" "$UNIT"
fi

if grep -q ORGANISM_DNA_VARIANT "$UNIT" 2>/dev/null; then
  sed -i 's/^Environment=ORGANISM_DNA_VARIANT=.*/Environment=ORGANISM_DNA_VARIANT=baby/' "$UNIT"
else
  sed -i '/ORGANISM_BABY_STATE/a Environment=ORGANISM_DNA_VARIANT=baby' "$UNIT"
fi

systemctl daemon-reload
systemctl restart organism-nursery.service
sleep 3
systemctl --no-pager status organism-nursery.service || true

echo ""
echo "==> Verifica"
curl -sf http://127.0.0.1:8765/organism/api/baby/state | python3 -c "
import json,sys
d=json.load(sys.stdin)
s=d.get('stats',{})
print('OK — neuroni:', s.get('neurons'), 'species:', s.get('species'))
" || echo "Servizio non ancora risponde — attendi o controlla journalctl -u organism-nursery"
