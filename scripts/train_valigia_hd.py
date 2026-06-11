#!/usr/bin/env python3
"""Training visivo HD — solo valigie, immagini reali Wikimedia fino a 1080p.

Uso sul server:
  cd /opt/mind-runtime && .venv/bin/python scripts/train_valigia_hd.py

Via API (nursery attivo):
  .venv/bin/python scripts/train_valigia_hd.py http://127.0.0.1:8765/organism
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = sys.argv[1] if len(sys.argv) > 1 else ""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_local() -> dict:
    from organism.autonomous.baby_agent import BabyAgent
    from organism.autonomous.baby_store import BabyStore

    agent = BabyAgent(store_path=str(ROOT / "data" / "baby_state.json"))
    store = BabyStore(str(ROOT / "data" / "baby_state.json"))
    if store.exists():
        store.load(agent)
    agent.composer.bind_semantic(agent.semantic)
    log("purge + training valigia HD…")
    result = agent.run_valigia_hd_curriculum(purge=True, per_query=8, pause_s=0.3)
    protos = len(agent.visual_binder._prototypes.get("valigia", []))
    log(f"fatto: taught={result.get('taught')} prototypes={protos}")
    return result


def train_api(base: str) -> dict:
    body = json.dumps({"purge": True, "per_query": 8}).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/api/baby/valigia-hd-curriculum",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    if BASE:
        log(f"API {BASE}")
        result = train_api(BASE)
    else:
        result = train_local()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
