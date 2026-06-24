#!/usr/bin/env python3
"""GPU capacity benchmark — how many neurons/synapses can we run at 30 FPS?"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from organism.brain.gpu_engine import GPUBrainEngine
from organism.brain.architect import BrainArchitect


def run_benchmark(
    *,
    neurons: int = 22800,
    synapses_per_neuron: int = 50,
    scales: int = 4,
) -> None:
    engine = GPUBrainEngine(use_gpu=True)
    print(f"Backend: {engine.backend} | Device: {engine.device}")
    print()

    # Build actual brain for reference
    architect = BrainArchitect()
    brain = architect.build()
    actual = engine.estimate_capacity_for_brain(brain.neuron_count, brain.synapse_count)
    print("── Cervello biologico 23.8k (reale) ──")
    print(f"  Neuroni:   {brain.neuron_count:,}")
    print(f"  Sinapsi:   {brain.synapse_count:,}")
    print(f"  Tick:      {actual.propagate_ms:.2f} ms ({actual.ticks_per_second:.0f} tps)")
    print(f"  Stima max: {actual.max_neurons_estimate:,} neuroni @ 30 FPS")
    print()

    print("── Scale test ──")
    results = []
    for i in range(scales):
        n = neurons * (10 ** i)
        if n > 50_000_000:
            break
        r = engine.benchmark(n, synapses_per_neuron=synapses_per_neuron)
        results.append(r)
        print(
            f"  {r.neurons:>12,} neurons | {r.synapses:>12,} synapses | "
            f"{r.propagate_ms:>7.2f} ms/tick | {r.ticks_per_second:>8.0f} tps | "
            f"max~{r.max_neurons_estimate:,}"
        )

    out = {
        "backend": engine.backend,
        "device": engine.device,
        "biological_brain": actual.to_dict(),
        "scales": [r.to_dict() for r in results],
    }
    print()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--neurons", type=int, default=22800)
    p.add_argument("--synapses-per", type=int, default=50)
    p.add_argument("--scales", type=int, default=4)
    a = p.parse_args()
    run_benchmark(neurons=a.neurons, synapses_per_neuron=a.synapses_per, scales=a.scales)
