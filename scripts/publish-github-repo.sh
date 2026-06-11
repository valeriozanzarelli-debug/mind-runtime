#!/usr/bin/env bash
# Pubblica questo tree su github.com/valeriozanzarelli-debug/mind-runtime
# Richiede: gh auth login (proprietario account)
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v gh >/dev/null; then
  echo "Installa GitHub CLI: https://cli.github.com/"
  exit 1
fi

if git remote get-url origin &>/dev/null; then
  url=$(git remote get-url origin)
  if [[ "$url" != *"mind-runtime"* ]]; then
    git remote rename origin ink-app-export 2>/dev/null || true
    git remote add origin https://github.com/valeriozanzarelli-debug/mind-runtime.git
  fi
else
  git remote add origin https://github.com/valeriozanzarelli-debug/mind-runtime.git
fi

if gh repo view valeriozanzarelli-debug/mind-runtime &>/dev/null; then
  git push -u origin main
else
  gh repo create valeriozanzarelli-debug/mind-runtime \
    --public --source=. --remote=origin --push \
    --description "ORGANISM cognitive runtime — mind + baby agent + nursery"
fi

echo "OK: https://github.com/valeriozanzarelli-debug/mind-runtime"
