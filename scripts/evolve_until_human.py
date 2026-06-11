#!/usr/bin/env python3
"""Loop infinito: probe → evolve DNA → train — finché pensiero/discorso umano."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organism.autonomous.baby_agent import BabyAgent
from organism.autonomous.baby_store import baby_state_path
from organism.cognition.discourse import is_babble
from organism.cognition.human_thought import thought_coherence

LOG = Path("/tmp/organism_evolve_until_human.log")
TARGET_SCORE = float(os.environ.get("ORGANISM_EVOLVE_TARGET", "0.78"))
MAX_CYCLES = int(os.environ.get("ORGANISM_EVOLVE_MAX_CYCLES", "0"))  # 0 = infinito


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def cycle_score(agent: BabyAgent) -> dict:
    m = agent._probe_metrics()
    ev = agent.dna_evolver.evaluate_metrics(m)
    mature = 0
    for _q, spoke in m["probe_outputs"]:
        if spoke and not is_babble(spoke, min_words=6):
            mature += 1
    ev["mature_probes"] = mature
    ev["mature_ratio"] = mature / max(1, len(m["probe_outputs"]))
    return {**m, **ev}


def main() -> None:
    os.environ.setdefault("ORGANISM_DNA_VARIANT", "genesis")
    agent = BabyAgent(store_path=str(baby_state_path()))
    if not agent._born:
        log("=== birth genesis (vuoto ma capace) ===")
        agent.birth_genesis()
    elif agent.org and agent.org.dna.genome.get("baby", {}).get("genesis") is not True:
        log("=== migrazione a genesis ===")
        agent.birth_genesis()

    n = 0
    while MAX_CYCLES == 0 or n < MAX_CYCLES:
        n += 1
        log(f"--- ciclo {n} ---")
        sc = cycle_score(agent)
        log(json.dumps({k: sc[k] for k in ("score", "mature_ratio", "avg_words", "thought_coherence") if k in sc}))

        if sc.get("score", 0) >= TARGET_SCORE and sc.get("mature_ratio", 0) >= 0.8:
            log(f"TARGET raggiunto score={sc['score']}")
            agent.export_genesis_dna()
            break

        ev = agent.evolve_dna(force=sc.get("score", 0) < 0.55)
        log(f"DNA evolve: {json.dumps(ev, ensure_ascii=False)[:200]}")

        agent.absorb_vocabulary(
            ["osservo", "collego", "ricordo", "capisco", "spiego", "rifletto", "sento", "penso"],
            boost=0.6,
        )
        for when, say in [
            ("ciao", "ciao ti ascolto e rispondo con attenzione"),
            ("cosa pensi", "penso a quello che percepisco in questo momento"),
            ("chi sei", "sono un organismo che impara dal mondo"),
        ]:
            agent.teach_dialogue(when, say)

        agent.brain_pulse_tick(persist=False)
        time.sleep(0.5)
        if n % 5 == 0:
            agent._persist()
            agent.export_genesis_dna()

    log("=== export DNA genesis finale ===")
    exp = agent.export_genesis_dna()
    log(json.dumps({k: exp[k] for k in ("ok", "generation", "version", "path")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
