"""CLI — demo e test interattivo al volo."""

from __future__ import annotations

import argparse
import json
import sys

from mind.runtime import MindRuntime, demo_scenarios
from mind.types import CostLevel, Cue, CueKind


def main() -> None:
    parser = argparse.ArgumentParser(description="MIND cognitive runtime")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="Esegui scenari precaricati")

    ip = sub.add_parser("interactive", help="Test al volo — inserisci cue")
    ip.add_argument("--visual", action="store_true", help="Cue visiva (forme+)")

    tp = sub.add_parser("think", help="Singola cue da riga di comando")
    tp.add_argument("text", help="Testo cue")
    tp.add_argument("--visual", action="store_true")
    tp.add_argument("--cost", choices=["low", "medium", "high"])
    tp.add_argument("--resonate-with", default=None)

    args = parser.parse_args()
    rt = MindRuntime.load_seed()

    if args.cmd == "demo":
        _run_demo(rt)
    elif args.cmd == "interactive":
        _run_interactive(rt, visual=args.visual)
    elif args.cmd == "think":
        kind = CueKind.VISUAL if args.visual else CueKind.TEXT
        cost = CostLevel(args.cost) if args.cost else None
        cue = Cue(kind=kind, value=args.text, meta={"human": True} if not args.visual else {})
        print(rt.think_json(cue, cost_override=cost, resonate_with=args.resonate_with))


def _run_demo(rt: MindRuntime) -> None:
    print("=" * 60)
    print("MIND RUNTIME — demo scenari")
    print("=" * 60)
    for name, cue, kwargs in demo_scenarios():
        print(f"\n▶ {name}")
        print(f"  cue: {cue.value[:70]}{'...' if len(cue.value) > 70 else ''}")
        result = rt.think(cue, **kwargs)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print("\n" + "=" * 60)


def _run_interactive(rt: MindRuntime, *, visual: bool) -> None:
    print("MIND interactive — esci con quit / exit")
    print("Modalità:", "VISUAL" if visual else "TEXT")
    print("Esempio visual: quadrato+cerchio,triangolo+cerchio,rettangolo+")
    while True:
        try:
            line = input("\nmind> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        kind = CueKind.VISUAL if visual or "+," in line or "+;" in line else CueKind.TEXT
        cue = Cue(kind=kind, value=line, meta={"human": kind == CueKind.TEXT})
        resonate = None
        if "|" in line:
            line, resonate = [p.strip() for p in line.split("|", 1)]
            cue = Cue(kind=kind, value=line, meta={"human": kind == CueKind.TEXT})
        print(rt.think_json(cue, resonate_with=resonate))


if __name__ == "__main__":
    main()
