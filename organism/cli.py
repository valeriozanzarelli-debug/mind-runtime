"""ORGANISM CLI."""

from __future__ import annotations

import argparse
import os
import sys

from organism.brain.architect import BrainArchitect
from organism.brain.runtime import BrainRuntime
from organism.brain.gpu_engine import GPUBrainEngine


def cmd_nursery(args: argparse.Namespace) -> None:
    from organism.nursery.server import NurseryServer

    NurseryServer(host=args.host, port=args.port, seed=args.seed).start()


def cmd_build(args: argparse.Namespace) -> None:
    architect = BrainArchitect(seed=args.seed)
    summary = architect.summary()
    print("🧠 Building biological brain...")
    t0 = __import__("time").perf_counter()
    brain = architect.build()
    elapsed = __import__("time").perf_counter() - t0
    stats = brain.stats()
    print(f"   Neuroni:  {stats['neurons']:,}")
    print(f"   Sinapsi:  {stats['synapses']:,}")
    print(f"   Regioni:  {len(stats['regions'])}")
    print(f"   Tempo:    {elapsed:.2f}s")
    print(f"   Genome:   {summary['genome_version']}")


def cmd_capacity(args: argparse.Namespace) -> None:
    from scripts.gpu_capacity import run_benchmark

    run_benchmark(
        neurons=args.neurons,
        synapses_per_neuron=args.synapses_per,
        scales=args.scales,
    )


def cmd_tick(args: argparse.Namespace) -> None:
    runtime = BrainRuntime.create(seed=args.seed)
    runtime.birth()
    runtime.perceive_text(args.text)
    for i in range(args.ticks):
        result = runtime.tick()
        print(f"tick {i+1}: Φ={result['consciousness']['phi']:.3f} "
              f"active={result['propagate']['active']} "
              f"DA={result['dopamine']['level']:.3f}")
    if args.chat:
        print(f"output: {runtime.chat(args.text)['output']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="organism", description="Biological brain runtime")
    sub = parser.add_subparsers(dest="command")

    p_nursery = sub.add_parser("nursery", help="Start HTTP nursery server")
    p_nursery.add_argument("--host", default=os.environ.get("ORGANISM_HOST", "127.0.0.1"))
    p_nursery.add_argument("--port", type=int, default=int(os.environ.get("ORGANISM_PORT", "8765")))
    p_nursery.add_argument("--seed", type=int, default=42)
    p_nursery.set_defaults(func=cmd_nursery)

    p_build = sub.add_parser("build", help="Build brain and print stats")
    p_build.add_argument("--seed", type=int, default=42)
    p_build.set_defaults(func=cmd_build)

    p_cap = sub.add_parser("capacity", help="GPU/CPU capacity benchmark")
    p_cap.add_argument("--neurons", type=int, default=22800)
    p_cap.add_argument("--synapses-per", type=int, default=50)
    p_cap.add_argument("--scales", type=int, default=4, help="Number of scale steps")
    p_cap.set_defaults(func=cmd_capacity)

    p_tick = sub.add_parser("tick", help="Run brain ticks on text input")
    p_tick.add_argument("--text", default="ciao mondo")
    p_tick.add_argument("--ticks", type=int, default=5)
    p_tick.add_argument("--chat", action="store_true")
    p_tick.add_argument("--seed", type=int, default=42)
    p_tick.set_defaults(func=cmd_tick)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
