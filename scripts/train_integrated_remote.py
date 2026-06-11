#!/usr/bin/env python3
"""Training Fase 3 via API nursery — funziona anche prima del deploy del nuovo codice.

Usa teach-dialogue, read, sense, self-hear, mega-curriculum, sleep.
Benchmark 100 prompt con scoring lato client (fluency_benchmark).

  python3 scripts/train_integrated_remote.py https://inkconscius.eu/organism
"""

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

from organism.teaching.fluency_benchmark import run_benchmark, score_response, aggregate_scores, standard_prompts
from organism.teaching.story_curriculum import dialogue_chains, long_story_chunks
from organism.teaching.syntax_curriculum import curriculum_sentences
from organism.teaching.corpus import REASONING, PHILOSOPHY, STORIES_EXTENDED

BASE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ORGANISM_URL", "https://inkconscius.eu/organism")
LOG = Path(os.environ.get("TRAIN_LOG", "/tmp/organism_train_remote.log"))
CYCLES = int(os.environ.get("TRAIN_CYCLES", "500"))
BENCHMARK_EVERY = int(os.environ.get("BENCHMARK_EVERY", "100"))
BENCHMARK_LIMIT = int(os.environ.get("BENCHMARK_LIMIT", "100"))
SYNTAX_BATCH = int(os.environ.get("SYNTAX_BATCH", "80"))


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def post(path: str, body: dict | None = None, *, timeout: float = 180.0) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{BASE.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            last = e
            time.sleep(2**attempt)
    raise last  # type: ignore[misc]


def get_state() -> dict:
    with urllib.request.urlopen(f"{BASE.rstrip('/')}/api/baby/state", timeout=60) as resp:
        return json.loads(resp.read().decode())


def teach_syntax_batch(*, limit: int = SYNTAX_BATCH, repeats: int = 2) -> int:
    sents = curriculum_sentences()[:limit]
    ok = 0
    for sent in sents:
        for _ in range(repeats):
            try:
                r = post("/api/baby/teach-dialogue", {"when": sent[:60], "say": sent, "kind": "speech"})
                if r.get("learned"):
                    ok += 1
            except Exception:
                pass
        try:
            post("/api/baby/read", {"text": sent})
        except Exception:
            pass
        time.sleep(0.08)
    return ok


def remote_sense(prompt: str) -> dict:
    return post("/api/baby/sense", {"text": prompt}, timeout=120.0)


def run_remote_benchmark(*, limit: int = 100) -> dict:
    return run_benchmark(remote_sense, limit=limit, pause_s=0.12)


def one_cycle(cycle: int) -> dict:
    import random

    rng = random.Random(cycle)
    actions: list[str] = []

    if cycle % 25 == 0:
        chunks = long_story_chunks()
        chunk = chunks[cycle % len(chunks)]
        post("/api/baby/read", {"text": chunk})
        actions.append("story")

    if cycle % 30 == 0:
        pairs = STORIES_EXTENDED + REASONING + PHILOSOPHY
        when, say = pairs[cycle % len(pairs)]
        post("/api/baby/teach-dialogue", {"when": when, "say": say})
        actions.append("dialogue")

    if cycle % 50 == 0:
        when, say = dialogue_chains()[cycle % len(dialogue_chains())]
        post("/api/baby/teach-dialogue", {"when": when, "say": say})
        actions.append("chain")

    if cycle % 150 == 0:
        try:
            post("/api/baby/mega-curriculum", {"limit": 25, "pause_s": 0.2})
            actions.append("mega")
        except Exception as e:
            actions.append(f"mega_err:{e!s:.30}")

    if cycle % 200 == 0:
        post("/api/baby/sleep", {})
        actions.append("sleep")

    q = rng.choice(["cosa pensi", "cosa vedi", "chi sei", "perché impari", "io sto imparando"])
    m = remote_sense(q).get("moment") or {}
    spoke = str(m.get("spoke", ""))
    if spoke and len(spoke) > 8:
        try:
            post("/api/baby/self-hear", {"text": spoke[:200]})
            actions.append("self-hear")
        except Exception:
            pass
    return {"cycle": cycle, "probe": q, "words": len(spoke.split()), "actions": actions}


def main() -> None:
    LOG.write_text(f"=== train_integrated_remote @ {BASE} cycles={CYCLES} ===\n")
    st = get_state()
    log(f"stato: born={st.get('born')} neuroni={st.get('neurons')} words={st.get('words_known')}")

    if not st.get("born"):
        post("/api/baby/birth", {})

    log("=== plasticità training ===")
    post("/api/baby/plasticity", {"rate": 0.035, "decay": 0.0005})

    log(f"=== syntax batch ({SYNTAX_BATCH} frasi) ===")
    n = teach_syntax_batch()
    log(f"syntax learned flags: {n}")

    log("=== benchmark iniziale ===")
    try:
        bm0 = run_remote_benchmark(limit=min(20, BENCHMARK_LIMIT))
        log(json.dumps(bm0.get("summary", {}), ensure_ascii=False))
    except Exception as e:
        log(f"benchmark iniziale skip: {e!s:.80}")

    best = 0.0
    for cycle in range(1, CYCLES + 1):
        try:
            step = one_cycle(cycle)
        except Exception as e:
            log(f"ciclo {cycle} ERR {e!s:.100}")
            continue
        if cycle % 50 == 0:
            log(f"ciclo {cycle}/{CYCLES} {step}")

        if cycle % BENCHMARK_EVERY == 0:
            log(f"\n######## BENCHMARK @ {cycle} ########")
            bm = run_remote_benchmark(limit=BENCHMARK_LIMIT)
            s = bm.get("summary", {})
            coh = float(s.get("coherent_ratio", 0))
            best = max(best, coh)
            log(
                f"coherent={coh:.1%} mature={s.get('mature_ratio', 0):.1%} "
                f"avg_words={s.get('avg_words')} babble={s.get('babble_ratio', 0):.1%}"
            )
            for row in bm.get("worst", [])[:2]:
                log(f"  worst: {row.get('prompt')} → {str(row.get('spoke',''))[:50]}")
            if coh >= 0.9:
                log("TARGET 90% raggiunto — stop anticipato")
                break

    log("=== benchmark finale ===")
    final = run_remote_benchmark(limit=BENCHMARK_LIMIT)
    log(json.dumps(final.get("summary", {}), ensure_ascii=False))
    post("/api/baby/plasticity", {"rate": 0.015, "decay": 0.0005})
    log(f"=== fine best_coherent={best:.1%} ===")


if __name__ == "__main__":
    main()
