#!/usr/bin/env python3
"""Retina GPU locale — benchmark e diagnostica (Windows + NVIDIA CUDA)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organism.brain.consciousness_probe import ConsciousnessProbe
from organism.brain.gpu_backend import gpu_info
from organism.brain.retina_cortex import create_retina_cortex


def _bright_spot(size: int, cx: int, cy: int, r: int = 6) -> list[list[int]]:
    grid = [[0] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                grid[y][x] = 230
    return grid


def bench(width: int, height: int, device: str, steps: int, pulses: int) -> dict:
    t0 = time.perf_counter()
    cortex = create_retina_cortex(width, height, device=device)
    birth_ms = (time.perf_counter() - t0) * 1000

    grid = _bright_spot(min(width, height), min(width, height) // 2, min(width, height) // 3)
    cortex.inject_pixels(grid)

    times: list[float] = []
    probe = ConsciousnessProbe()
    last_snap = None
    for _ in range(pulses):
        t1 = time.perf_counter()
        cortex.propagate(steps=steps)
        last_snap = probe.read(cortex, sensory_tags=["VIS:local"], pressure=0.25)
        if HAS_TORCH_SYNC and cortex.uses_gpu:
            import torch

            torch.cuda.synchronize()
        times.append((time.perf_counter() - t1) * 1000)

    return {
        "width": width,
        "height": height,
        "neurons": cortex.neuron_count,
        "device": cortex.stats()["backend"],
        "uses_gpu": cortex.uses_gpu,
        "birth_ms": round(birth_ms, 2),
        "pulse_ms_avg": round(sum(times) / len(times), 3),
        "pulse_ms_max": round(max(times), 3),
        "conscious": last_snap.conscious if last_snap else False,
        "ignition": round(last_snap.ignition, 4) if last_snap else 0,
        "focus": last_snap.focus.to_dict() if last_snap and last_snap.focus else None,
        **cortex.stats(),
    }


try:
    import torch

    HAS_TORCH_SYNC = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_SYNC = False


def main() -> None:
    p = argparse.ArgumentParser(description="Retina cortex GPU locale (Windows CUDA)")
    p.add_argument("--info", action="store_true", help="Mostra info GPU e esci")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu | numpy")
    p.add_argument("--steps", type=int, default=3, help="propagate steps per pulse")
    p.add_argument("--pulses", type=int, default=30)
    p.add_argument("--preset", choices=["hd", "fullhd", "4k", "baby"], default=None)
    args = p.parse_args()

    if args.info:
        print(json.dumps(gpu_info(), indent=2))
        return

    presets = {
        "baby": (320, 256),
        "hd": (1024, 768),
        "fullhd": (1920, 1080),
        "4k": (3840, 2160),
    }
    w, h = presets.get(args.preset, (args.width, args.height))

    print(f"Retina {w}×{h} = {w * h:,} neuroni · device={args.device}")
    info = gpu_info()
    if args.device in ("auto", "cuda") and not info.get("cuda_available"):
        print("ATTENZIONE: CUDA non disponibile — userà CPU/numpy", file=sys.stderr)
        if info.get("install_hint"):
            print(info["install_hint"], file=sys.stderr)

    result = bench(w, h, args.device, args.steps, args.pulses)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
