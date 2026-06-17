"""ORGANISM CLI — perceive → think → express → learn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from organism.integrations.ink_api import InkApiBridge, WaMessage
from organism.runtime import OrganismRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="ORGANISM cognitive runtime")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="Brain stats from DNA growth")

    demo = sub.add_parser("demo", help="Full multimodal demo scenarios")
    demo.add_argument("--variant", choices=["base", "studio"], default="studio")

    live = sub.add_parser("live", help="Single live cycle with learning")
    live.add_argument("--text", default=None)
    live.add_argument("--shapes", default=None)
    live.add_argument("--tone", type=float, default=None)
    live.add_argument("--modality", choices=["speech", "song", "text", "motion", "full"], default="speech")
    live.add_argument("--resonate-with", default=None)
    live.add_argument("--cost", choices=["low", "medium", "high"], default=None)
    live.add_argument("--no-learn", action="store_true")

    replay = sub.add_parser("replay", help="Batch training episodes")
    replay.add_argument("--file", required=True, help="JSON file: [{input:{text:...}}, ...]")

    save = sub.add_parser("save", help="Persist brain + memory state")
    save.add_argument("--path", default=None)

    load = sub.add_parser("load", help="Load persisted state into fresh organism")
    load.add_argument("--path", default=None)

    wa = sub.add_parser("wa", help="Mock WhatsApp message through ink-api bridge")
    wa.add_argument("--text", required=True)
    wa.add_argument("--thread", default="wa_demo_1")

    nursery = sub.add_parser("nursery", help="Dashboard web — vedi pensieri e connessioni")
    nursery.add_argument("--host", default="127.0.0.1")
    nursery.add_argument("--port", type=int, default=8765)
    nursery.add_argument("--browser", action="store_true")

    sub.add_parser("sleep", help="Prune + consolidate")

    viz = sub.add_parser("brain", help="Visualize brain sample")
    viz.add_argument("--max-nodes", type=int, default=30)

    retina = sub.add_parser("retina", help="Retina GPU locale — milioni di neuroni-pixel")
    retina.add_argument("--info", action="store_true", help="Diagnostica GPU")
    retina.add_argument("--width", type=int, default=1024)
    retina.add_argument("--height", type=int, default=768)
    retina.add_argument("--device", default="auto", help="auto | cuda | cpu | numpy")
    retina.add_argument("--preset", choices=["baby", "hd", "fullhd"], default="hd")
    retina.add_argument("--pulses", type=int, default=15)

    args = parser.parse_args()

    if args.cmd == "stats":
        org = OrganismRuntime.studio_assistant()
        print(json.dumps(org.stats, indent=2))
    elif args.cmd == "demo":
        _run_demo(_load(args))
    elif args.cmd == "live":
        org = OrganismRuntime.studio_assistant()
        print(org.live_json(_input_from_args(args), **_kwargs_from_args(args, learn=not args.no_learn)))
    elif args.cmd == "replay":
        org = OrganismRuntime.studio_assistant()
        episodes = json.loads(Path(args.file).read_text(encoding="utf-8"))
        print(json.dumps(org.replay(episodes), ensure_ascii=False, indent=2))
    elif args.cmd == "save":
        org = OrganismRuntime.studio_assistant()
        p = org.save_state(args.path)
        print(json.dumps({"saved": str(p), "stats": org.stats}, indent=2))
    elif args.cmd == "load":
        org = OrganismRuntime.studio_assistant()
        info = org.load_state(args.path)
        print(json.dumps({**info, "stats": org.stats}, indent=2))
    elif args.cmd == "wa":
        org = OrganismRuntime.studio_assistant()
        bridge = InkApiBridge(mock=True)
        reply = bridge.live_from_message(org, WaMessage.from_mock(args.text, thread_id=args.thread))
        print(json.dumps(reply.__dict__, ensure_ascii=False, indent=2))
    elif args.cmd == "nursery":
        from organism.nursery.server import NurseryServer
        NurseryServer(args.host, args.port).start(open_browser=args.browser)
    elif args.cmd == "sleep":
        org = OrganismRuntime.studio_assistant()
        print(json.dumps(org.sleep(), indent=2))
    elif args.cmd == "brain":
        org = OrganismRuntime.studio_assistant()
        print(org.brain_visualize(max_nodes=args.max_nodes))
    elif args.cmd == "retina":
        from organism.brain.gpu_backend import gpu_info
        from organism.brain.retina_cortex import create_retina_cortex
        from organism.brain.consciousness_probe import ConsciousnessProbe
        import time

        if args.info:
            print(json.dumps(gpu_info(), indent=2))
        else:
            presets = {"baby": (320, 256), "hd": (1024, 768), "fullhd": (1920, 1080)}
            w, h = presets.get(args.preset, (args.width, args.height))
            cortex = create_retina_cortex(w, h, device=args.device)
            grid = [[0] * w for _ in range(h)]
            cx, cy = w // 2, h // 3
            for y in range(h):
                for x in range(w):
                    if (x - cx) ** 2 + (y - cy) ** 2 < 36:
                        grid[y][x] = 220
            cortex.inject_pixels(grid)
            probe = ConsciousnessProbe()
            times = []
            snap = None
            for _ in range(args.pulses):
                t0 = time.perf_counter()
                cortex.propagate(steps=3)
                snap = probe.read(cortex, sensory_tags=["VIS:scene"], pressure=0.3)
                times.append(time.perf_counter() - t0)
            out = {
                **cortex.stats(),
                "pulse_ms_avg": round(1000 * sum(times) / len(times), 2),
                "consciousness": snap.to_dict() if snap else {},
            }
            print(json.dumps(out, indent=2))


def _load(args) -> OrganismRuntime:
    if getattr(args, "variant", "studio") == "studio":
        return OrganismRuntime.studio_assistant()
    return OrganismRuntime()


def _input_from_args(args) -> dict:
    data: dict = {}
    if args.text:
        data["text"] = args.text
    if args.shapes:
        data["shapes"] = args.shapes
    if args.tone:
        data["tone_hz"] = args.tone
    return data


def _kwargs_from_args(args, **extra) -> dict:
    kwargs = dict(extra)
    if args.resonate_with:
        kwargs["resonate_with"] = args.resonate_with
    if args.cost:
        from mind.types import CostLevel
        kwargs["cost_override"] = CostLevel(args.cost)
    return kwargs


def _run_demo(org: OrganismRuntime) -> None:
    scenarios = [
        ("Puzzle forme", {"shapes": "quadrato+cerchio,triangolo+cerchio,rettangolo+"}, "text"),
        ("Lampadina + learn", {"text": "la lampadina non si accende"}, "speech"),
        ("WA prenotazione", {"text": "Ciao, vorrei prenotare per giovedì"}, "speech"),
        (
            "Cliente diffidente",
            {"text": "cliente whatsapp diffidente chiede preventivo braccio"},
            "speech",
        ),
        ("Full multimodal", {"text": "preventivo tattoo realistico braccio"}, "full"),
    ]
    print("=" * 60)
    print(f"ORGANISM demo — {org.stats}")
    print("=" * 60)
    for name, data, mod in scenarios:
        print(f"\n▶ {name}")
        kwargs = {}
        if "diffidente" in data.get("text", ""):
            kwargs["resonate_with"] = "cliente marzo diffidente whatsapp"
        print(org.live_json(data, output_modality=mod, **kwargs))
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
