#!/usr/bin/env python3
"""Benchmark FPS — Mindruntime GPU V2 @ 256×256."""

from __future__ import annotations

import argparse
import time

import numpy as np

from mindruntime.cuda_util import cuda_info
from mindruntime.gpu_engine_v2 import BrainEngineV2


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark GPU physics V2 FPS")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    info = cuda_info()
    print(f"CUDA: {info}")
    engine = BrainEngineV2(width=args.width, height=args.height, seed=args.seed)
    frame = np.random.rand(args.height, args.width, 3).astype(np.float32) * 0.5 + 0.2

    for _ in range(args.warmup):
        engine.step(frame)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        engine.step(frame)
    if engine.uses_cuda:
        from numba import cuda

        cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    fps = args.steps / elapsed if elapsed > 0 else 0.0
    ms = (elapsed / args.steps) * 1000.0
    print(f"Backend: {engine.stats.backend}")
    print(f"Resolution: {args.width}×{args.height}")
    print(f"Steps: {args.steps}  elapsed: {elapsed:.3f}s")
    print(f"FPS sustained: {fps:.1f}  ({ms:.2f} ms/step)")
    print(f"Order R: {engine.stats.order_parameter:.3f}  coherence: {engine.stats.mean_coherence:.3f}")


if __name__ == "__main__":
    main()
