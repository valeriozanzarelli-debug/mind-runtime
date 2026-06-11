#!/usr/bin/env python3
"""Training integrato Fase 3 — 10k cicli, storie lunghe, motor loop, benchmark ogni 1k.

Uso locale (server):
  cd /opt/mind-runtime
  PYTHONPATH=. python3 scripts/train_integrated.py

Variabili:
  TRAIN_CYCLES=10000       cicli totali (default 10000)
  BENCHMARK_EVERY=1000       benchmark ogni N cicli
  ORGANISM_DNA_VARIANT=genesis
  TRAIN_LOG=/tmp/organism_train_integrated.log
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = Path(os.environ.get("TRAIN_LOG", "/tmp/organism_train_integrated.log"))
CYCLES = int(os.environ.get("TRAIN_CYCLES", "10000"))
BENCHMARK_EVERY = int(os.environ.get("BENCHMARK_EVERY", "1000"))
BENCHMARK_LIMIT = int(os.environ.get("BENCHMARK_LIMIT", "100"))
TARGET_COHERENT = float(os.environ.get("TARGET_COHERENT", "0.90"))


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main() -> None:
    from organism.autonomous.baby_agent import BabyAgent
    from organism.autonomous.baby_store import baby_state_path

    os.environ.setdefault("ORGANISM_DNA_VARIANT", "genesis")
    agent = BabyAgent(store_path=str(baby_state_path()))

    if not agent._born:
        log("=== birth genesis ===")
        agent.birth_genesis()
    elif agent.org and agent.org.dna.genome.get("baby", {}).get("genesis") is not True:
        log("=== migrazione genesis ===")
        agent.birth_genesis()
    else:
        log("=== resume stato esistente ===")
        agent._apply_cognition_config()
        agent._start_thought_loop()

    log("=== bootstrap syntax Layer 2 ===")
    boot = agent.train_syntax_curriculum(repeats=3, absorb_limit=400)
    log(json.dumps(boot, ensure_ascii=False)[:300])

    log(f"=== train_integrated cycles={CYCLES} benchmark_every={BENCHMARK_EVERY} ===")
    best_coherent = 0.0
    t_start = time.time()

    for cycle in range(1, CYCLES + 1):
        try:
            step = agent.train_integrated_cycle(cycle)
        except Exception as e:
            log(f"ciclo {cycle} ERR {e!s:.120}")
            continue

        if cycle % 100 == 0:
            log(
                f"ciclo {cycle}/{CYCLES} probe={step.get('probe')} "
                f"spoke={step.get('spoke_words')}w actions={len(step.get('actions', []))}"
            )

        if cycle % BENCHMARK_EVERY == 0:
            log(f"\n######## BENCHMARK @ ciclo {cycle} ########")
            agent.set_training_plasticity(rate=0.02, decay=0.0005)
            bm = agent.run_fluency_benchmark(limit=BENCHMARK_LIMIT, pause_s=0.1)
            summary = bm.get("summary", {})
            coherent = float(summary.get("coherent_ratio", 0))
            best_coherent = max(best_coherent, coherent)
            log(
                f"coherent={coherent:.1%} mature={summary.get('mature_ratio', 0):.1%} "
                f"avg_words={summary.get('avg_words')} svo={summary.get('svo_ratio', 0):.1%} "
                f"babble={summary.get('babble_ratio', 0):.1%}"
            )
            for row in bm.get("worst", [])[:3]:
                if hasattr(row, "to_dict"):
                    row = row.to_dict()
                log(f"  worst: {row.get('prompt')} → ({row.get('words')}w) {str(row.get('spoke',''))[:60]}")
            for row in bm.get("best", [])[:3]:
                if hasattr(row, "to_dict"):
                    row = row.to_dict()
                log(f"  best: {row.get('prompt')} → ({row.get('words')}w) {str(row.get('spoke',''))[:60]}")
            agent.set_training_plasticity(rate=0.015, decay=0.0005)
            agent._persist()

            if coherent >= TARGET_COHERENT:
                log(f"TARGET raggiunto: coherent_ratio={coherent:.1%} >= {TARGET_COHERENT:.0%}")
                break

        if cycle % 500 == 0:
            agent.export_genesis_dna()
            log(f"DNA export @ {cycle}")

    elapsed = time.time() - t_start
    log(f"\n=== fine training elapsed={elapsed/3600:.2f}h best_coherent={best_coherent:.1%} ===")
    final = agent.run_fluency_benchmark(limit=BENCHMARK_LIMIT, pause_s=0.08)
    log(json.dumps(final.get("summary", {}), ensure_ascii=False))
    agent._persist()
    agent.export_genesis_dna()


if __name__ == "__main__":
    main()
