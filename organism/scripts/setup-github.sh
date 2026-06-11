#!/usr/bin/env bash
# Crea il repo GitHub e fa il primo push (richiede gh auth login).
set -euo pipefail
cd "$(dirname "$0")/.."

if git remote get-url origin &>/dev/null; then
  echo "Remote origin già configurato:"
  git remote -v
  git push -u origin main
  exit 0
fi

gh repo create valeriozanzarelli-debug/mind-runtime \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "MIND + ORGANISM cognitive runtime — DNA-grown brain, sensory input, motor output"

echo "Repo pubblicato: https://github.com/valeriozanzarelli-debug/mind-runtime"
