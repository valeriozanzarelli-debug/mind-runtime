#!/usr/bin/env python3
"""Stress test — quanti neuroni/sinapsi regge il server con propagazione sparsa."""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organism.brain.oscillation import inject_wave
from organism.dna.interpreter import DNAInterpreter, merge_genomes, load_yaml
from organism.runtime import OrganismRuntime


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def grow_tier(tier: str, seed: int):
    base = load_yaml(ROOT / "organism" / "dna" / "organism_dna.yaml")
    tiers = {
        "default": {"scale": {"neuron_multiplier": 1.0, "connection_multiplier": 1.0}},
        "large": {
            "scale": {
                "neuron_multiplier": 50,
                "connection_multiplier": 0.4,
                "sparse_fan_out_cap": 12,
                "energy_budget_per_tick": 80_000,
            },
            "fractal_expansion": {"neurons_per_leaf": 8},
        },
        "mega": {},
    }
    if tier == "mega":
        return OrganismRuntime.mega(seed=seed).brain
    if tier == "giga":
        return OrganismRuntime.giga(seed=seed).brain
    if tier == "ultra":
        return OrganismRuntime.ultra(seed=seed).brain
    overlay = tiers.get(tier, tiers["default"])
    genome = merge_genomes(base, overlay)
    dna = DNAInterpreter()
    dna.genome = genome
    brain = dna.grow_brain(seed=seed)
    return brain


def bench(tier: str, seed: int, pulses: int) -> dict:
    gc.collect()
    t0 = time.perf_counter()
    factory = {
        "mega": OrganismRuntime.mega,
        "giga": OrganismRuntime.giga,
        "ultra": OrganismRuntime.ultra,
    }
    if tier in factory:
        brain = factory[tier](seed=seed).brain
    else:
        brain = grow_tier(tier, seed)
    birth_s = time.perf_counter() - t0
    mem_birth = _rss_mb()

    # stimolo locale — pochi neuroni sensoriali
    for n in brain.get_neurons("sensory", "text_semantic_encoder")[:12]:
        n.activation = 0.85
        brain._active.add(n.id)

    times: list[float] = []
    for i in range(pulses):
        inject_wave(brain, "think", tick=i, amplitude=0.7)
        t1 = time.perf_counter()
        brain.propagate(steps=2)
        times.append(time.perf_counter() - t1)
        if brain.plasticity and i % 5 == 0:
            brain.plasticity.apply_hebbian(brain, brain.tick)

    eff = brain.efficiency_stats()
    return {
        "tier": tier,
        "neurons": brain.neuron_count,
        "synapses": brain.synapse_count,
        "birth_s": round(birth_s, 2),
        "mem_mb": round(mem_birth, 1),
        "pulse_ms_avg": round(1000 * sum(times) / len(times), 2),
        "pulse_ms_max": round(1000 * max(times), 2),
        "active_ratio": eff["active_ratio"],
        "mean_fan_out": eff["mean_fan_out"],
        "energy_budget": eff["energy_budget"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark scala cervello ORGANISM")
    p.add_argument("--tier", choices=["default", "large", "mega", "giga", "ultra"], default="default")
    p.add_argument("--pulses", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all", action="store_true", help="Esegui default + large (+ mega se --allow-mega)")
    p.add_argument("--allow-mega", action="store_true", help="Includi tier mega (~1.5M neuroni)")
    p.add_argument("--allow-giga", action="store_true", help="Includi tier giga (~3M neuroni, ~8GB RAM)")
    p.add_argument("--allow-ultra", action="store_true", help="Includi tier ultra (~5M neuroni, ~14GB RAM)")
    args = p.parse_args()

    tiers = ["default", "large"]
    if args.all:
        if args.allow_mega:
            tiers.append("mega")
        if args.allow_giga:
            tiers.append("giga")
        if args.allow_ultra:
            tiers.append("ultra")
    else:
        tiers = [args.tier]

    print("tier\tneurons\tsynapses\tbirth_s\tmem_mb\tpulse_ms\tactive%\tfan_out")
    for tier in tiers:
        r = bench(tier, args.seed, args.pulses)
        print(
            f"{r['tier']}\t{r['neurons']}\t{r['synapses']}\t{r['birth_s']}\t"
            f"{r['mem_mb']}\t{r['pulse_ms_avg']}\t{r['active_ratio']*100:.3f}\t{r['mean_fan_out']}"
        )


if __name__ == "__main__":
    main()
