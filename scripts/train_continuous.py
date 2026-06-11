#!/usr/bin/env python3
"""Training continuo — linguaggio, mondo, oggetti, sonno, validazione."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"
LOG = Path("/tmp/organism_train_continuous.log")
CYCLE_SLEEP = int(os.environ.get("TRAIN_CYCLE_SLEEP", "300"))
MEGA_LIMIT = int(os.environ.get("MEGA_BATCH", "50"))


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def post(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def get_state() -> dict:
    with urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=120) as resp:
        return json.loads(resp.read().decode())


def benchmark(*, full: bool = False) -> dict:
    if full:
        try:
            return post(
                "/api/baby/fluency-benchmark",
                {"limit": int(os.environ.get("BENCHMARK_LIMIT", "100")), "pause_s": 0.1},
            )
        except Exception as e:
            log(f"fluency benchmark err: {e!s}"[:80])
    probes = [
        "ciao",
        "cosa vedi",
        "cosa pensi",
        "perché piove",
        "raccontami una storia",
        "scrivi funzione fibonacci",
    ]
    results: list[dict] = []
    mature = 0
    for q in probes:
        m = post("/api/baby/sense", {"text": q}).get("moment") or {}
        spoke = str(m.get("spoke", ""))
        words = len(spoke.split())
        if words >= 10:
            mature += 1
        results.append({"q": q, "words": words, "spoke": spoke[:120]})
        time.sleep(0.3)
    return {
        "mature_ratio": mature / max(1, len(probes)),
        "probes": results,
    }


def run_script(name: str) -> None:
    script = ROOT / "scripts" / name
    if not script.exists():
        script = ROOT / "organism" / "scripts" / name
    if not script.exists():
        log(f"skip {name} — non trovato")
        return
    log(f">>> {name}")
    subprocess.run(
        [sys.executable, str(script), BASE],
        check=False,
        timeout=7200,
    )


def main() -> None:
    cycle = 0
    log(f"=== train_continuous @ {BASE} ===")
    while True:
        cycle += 1
        log(f"\n######## CICLO {cycle} ########")
        st = get_state()
        if not st.get("born"):
            post("/api/baby/birth", {})
        log(
            f"stato: neuroni={st.get('neurons')} parole={st.get('words_known')} "
            f"dialoghi={len(st.get('dialogue_pairs', []))}"
        )

        post("/api/baby/plasticity", {"rate": 0.035})
        run_script("train_emergent_language.py")
        run_script("train_world.py")

        if cycle % 2 == 0:
            log(f"mega-curriculum batch {MEGA_LIMIT}")
            try:
                r = post("/api/baby/mega-curriculum", {"limit": MEGA_LIMIT, "pause_s": 0.25})
                log(json.dumps({k: r[k] for k in ("ok", "taught", "total") if k in r}))
            except Exception as e:
                log(f"mega err: {e!s}"[:80])

        if cycle % 3 == 0:
            post("/api/baby/code-curriculum", {})

        if cycle % 2 == 0:
            try:
                post("/api/baby/train-syntax", {"repeats": 2, "absorb_limit": 120})
            except Exception:
                pass

        if cycle % 5 == 0:
            post("/api/baby/sleep", {})
            post("/api/baby/evolve-dna", {"force": False})
            bm = benchmark(full=cycle % 10 == 0)
            if "summary" in bm:
                s = bm["summary"]
                log(
                    f"fluency coherent={s.get('coherent_ratio', 0):.0%} "
                    f"mature={s.get('mature_ratio', 0):.0%} avg_words={s.get('avg_words')}"
                )
            else:
                log(f"benchmark mature={bm.get('mature_ratio', 0):.0%}")
                for row in bm.get("probes", []):
                    log(f"  {row['q']}: ({row['words']}w) {row['spoke'][:70]}")

        post("/api/baby/plasticity", {"rate": 0.015})
        log(f"pausa {CYCLE_SLEEP}s…")
        time.sleep(CYCLE_SLEEP)


if __name__ == "__main__":
    main()
