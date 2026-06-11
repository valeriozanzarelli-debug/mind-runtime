#!/usr/bin/env python3
"""Training linguaggio emergente — corpus esteso, ripetizione 3-5x, motor loop."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organism.teaching.corpus import (
    HIGH_FREQ_BLOCKS,
    PHILOSOPHY,
    REASONING,
    STORIES_EXTENDED,
    CODE_DIALOGUES,
)

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"
LOG = Path("/tmp/organism_train_language.log")
REPEATS = int(os.environ.get("TRAIN_REPEATS", "4"))


def post(path: str, body: dict, *, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            last_err = e
            time.sleep(2**attempt)
    raise last_err  # type: ignore[misc]


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def teach_dialogue(when: str, say: str, *, kind: str = "speech", repeats: int = REPEATS) -> bool:
    learned = False
    for _ in range(repeats):
        r = post("/api/baby/teach-dialogue", {"when": when, "say": say, "kind": kind})
        learned = learned or bool(r.get("learned"))
    post("/api/baby/sense", {"text": when})
    time.sleep(0.12)
    return learned


def main() -> None:
    LOG.write_text(f"=== train_emergent_language @ {BASE} repeats={REPEATS} ===\n")
    try:
        post("/api/baby/plasticity", {"rate": 0.035, "decay": 0.0006})
        log("plasticità training: rate=0.035")
    except Exception as e:
        log(f"plasticity skip: {e!s}"[:60])

    total = ok = 0
    sections = [
        ("vocab", [(w, w) for block in HIGH_FREQ_BLOCKS for w in block]),
        ("storie", STORIES_EXTENDED),
        ("ragionamento", REASONING),
        ("filosofia", PHILOSOPHY),
    ]
    for cat, items in sections:
        log(f"--- {cat} ({len(items)}) ---")
        for when, say in items:
            total += 1
            if teach_dialogue(when, say):
                ok += 1
            if total % 50 == 0:
                post("/api/baby/sleep", {})
                log(f"consolidamento @ {total}")
        log(f"{cat} ok={ok}/{total}")

    log("--- codice ---")
    for when, say, kind in CODE_DIALOGUES:
        total += 1
        if teach_dialogue(when, say, kind=kind):
            ok += 1

    log("--- probe qualità ---")
    probes = [
        "cosa pensi",
        "raccontami pinocchio",
        "perché piove",
        "se piove cosa succede",
        "cos'è la coscienza",
        "scrivi funzione fibonacci",
    ]
    for q in probes:
        try:
            m = post("/api/baby/sense", {"text": q}).get("moment") or {}
            spoke = str(m.get("spoke", ""))
            words = len(spoke.split())
            inner = [ln for ln in m.get("consciousness_stream", []) if "pensiero:" in ln]
            log(f"  {q} → ({words}w) {spoke[:100]}")
            if inner:
                log(f"    {inner[0][:90]}")
        except Exception as e:
            log(f"  {q} ERR {e!s}"[:80])

    try:
        post("/api/baby/plasticity", {"rate": 0.015, "decay": 0.001})
    except Exception:
        pass

    st = json.loads(urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=120).read())
    log(
        f"=== fine {ok}/{total} | parole={st.get('words_known')} | "
        f"dialoghi={len(st.get('dialogue_pairs', []))} | "
        f"sinapsi={st.get('stats', {}).get('synapses', '?')} ==="
    )


if __name__ == "__main__":
    main()
