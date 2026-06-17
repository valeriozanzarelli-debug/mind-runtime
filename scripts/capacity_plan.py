#!/usr/bin/env python3
"""Calcola capacità massima — server 16 GB RAM + GPU locale 8 GB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organism.cognition.brain_budget import (
    analyze_brain,
    build_capacity_plan,
    human_comparison,
    recommend_gpu_resolution,
    recommend_graph_tier,
)
from organism.dna.interpreter import DNAInterpreter, load_yaml, merge_genomes, BASE_GENOME


def grow_variant(name: str, seed: int = 42):
    base = load_yaml(BASE_GENOME)
    variant_path = ROOT / "organism" / "dna" / "variants" / f"{name}.yaml"
    if not variant_path.exists():
        raise SystemExit(f"variante sconosciuta: {name}")
    genome = merge_genomes(base, load_yaml(variant_path))
    dna = DNAInterpreter()
    dna.genome = genome
    return dna.grow_brain(seed=seed)


def main() -> None:
    p = argparse.ArgumentParser(description="Piano capacità cervello delocalizzato")
    p.add_argument("--server-ram-gb", type=int, default=16)
    p.add_argument("--gpu-vram-gb", type=int, default=8)
    p.add_argument("--variant", default="mind_giga", help="profilo DNA da analizzare")
    p.add_argument("--grow", action="store_true", help="cresci il cervello (lento per giga)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    gw, gh, gpu_mb = recommend_gpu_resolution(args.gpu_vram_gb * 1024)
    tier = recommend_graph_tier(args.server_ram_gb)

    if args.grow:
        brain = grow_variant(args.variant)
        budget = analyze_brain(brain)
        plan = build_capacity_plan(
            graph_neurons=budget.total,
            graph_thinking=budget.thinking,
            graph_synapses=budget.synapses,
            gpu_w=gw,
            gpu_h=gh,
        )
    else:
        # stime senza boot pesante
        tinfo = tier["tiers"].get(args.variant, tier["tiers"].get("mind_giga", {}))
        n = int(tinfo.get("neurons", 4_000_000))
        plan = build_capacity_plan(
            graph_neurons=n,
            graph_thinking=int(n * 0.52),
            graph_synapses=int(n * 4.2),
            gpu_w=gw,
            gpu_h=gh,
        )
        budget = None

    human = human_comparison(plan.graph_thinking, plan.graph_neurons)
    out = {
        "hardware": {
            "server_ram_gb": args.server_ram_gb,
            "gpu_vram_gb": args.gpu_vram_gb,
            "recommended_dna_variant": tier["recommended_variant"],
            "analyzed_variant": args.variant,
        },
        "gpu_field": {
            "resolution": plan.gpu_resolution,
            "pixel_neurons": plan.gpu_pixels,
            "estimated_vram_mb": plan.gpu_ram_mb,
        },
        "graph_dna": plan.to_dict() if budget is None else {**plan.to_dict(), **budget.to_dict()},
        "total_effective_neurons": plan.total_effective_neurons,
        "human_comparison": human,
        "env_server": {
            "ORGANISM_DNA_VARIANT": args.variant,
            "ORGANISM_GPU_REMOTE": "http://IP-PC-LOCALE:8770",
            "ORGANISM_IMPULSE": "1",
        },
        "env_gpu_pc": {
            "ORGANISM_IMPULSE_W": gw,
            "ORGANISM_IMPULSE_H": gh,
            "ORGANISM_GPU_WORKER_DEVICE": "cuda",
        },
    }

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print("=== CERVELLO DELOCALIZZATO — piano capacità ===\n")
    print(f"Server RAM: {args.server_ram_gb} GB  →  variante consigliata: {tier['recommended_variant']}")
    print(f"GPU locale: {args.gpu_vram_gb} GB  →  campo impulsi {gw}×{gh} = {plan.gpu_pixels:,} neuroni-pixel (~{gpu_mb:.0f} MB VRAM)\n")
    if budget:
        print(f"Grafo {args.variant}: {budget.total:,} neuroni ({budget.thinking:,} pensiero = {100*budget.thinking_ratio:.1f}%)")
        print(f"  corpo/motorio: {budget.motor_body:,} ({100*budget.body_motor_ratio:.1f}%)  |  linguaggio: {budget.motor_speech:,}")
        print(f"  sinapsi: {budget.synapses:,}  |  RAM stimata: ~{plan.graph_ram_mb:.0f} MB")
    else:
        print(f"Grafo stimato ({args.variant}): ~{plan.graph_neurons:,} neuroni (~{plan.graph_thinking:,} pensiero)")
    print(f"\nTOTALE EFFETTIVO: {plan.total_effective_neurons:,} neuroni (grafo + GPU pixel)")
    print(f"\nCervello umano: {human['human_total_neurons']:,} totali, ma ~{human['human_cerebellum_skipped']:,} nel cerebellum (motorio) che noi NON simuliamo.")
    print(f"Pensiero umano stimato: ~{human['human_thinking_estimate']:,}  |  nostri neuroni pensiero: {human['our_thinking_neurons']:,}")
    print("\nAvvio GPU worker (PC locale):")
    print(f"  ORGANISM_IMPULSE_W={gw} ORGANISM_IMPULSE_H={gh} python3 -m organism.distributed.gpu_worker_server --port 8770")
    print("\nServer (nursery):")
    print(f"  ORGANISM_DNA_VARIANT={args.variant} ORGANISM_GPU_REMOTE=http://TUO-PC:8770")


if __name__ == "__main__":
    main()
