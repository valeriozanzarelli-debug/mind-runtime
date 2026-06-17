#!/usr/bin/env python3
"""Verifica CUDA + Numba — esegui su Windows con RTX 1060."""

from __future__ import annotations

import sys


def main() -> int:
    print("=" * 50)
    print("TEST CUDA + NUMBA — Mindruntime V2")
    print("=" * 50)

    try:
        from numba import cuda
        import numpy as np
    except ImportError as exc:
        print(f"❌ Dipendenze mancanti: {exc}")
        print("   pip install numba numpy")
        return 1

    if not cuda.is_available():
        print("❌ CUDA NON disponibile")
        print("Verifica:")
        print("  1. Driver NVIDIA aggiornati (nvidia-smi)")
        print("  2. CUDA Toolkit installato (nvcc --version)")
        print("  3. GPU compatibile con Numba")
        return 1

    dev = cuda.get_current_device()
    name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
    print(f"✅ CUDA disponibile")
    print(f"✅ Device: {name}")
    print(f"✅ Compute capability: {dev.compute_capability}")

    @cuda.jit
    def test_kernel(arr):
        x = cuda.grid(1)
        if x < arr.size:
            arr[x] = x * 2.0

    arr_host = np.zeros(1000, dtype=np.float32)
    arr_device = cuda.to_device(arr_host)
    test_kernel[10, 100](arr_device)
    cuda.synchronize()
    result = arr_device.copy_to_host()
    assert result[10] == 20.0, f"expected 20.0 got {result[10]}"
    print("✅ Test kernel PASSED")

    from mindruntime.cuda_util import cuda_info
    from mindruntime.gpu_engine_v2 import BrainEngineV2

    info = cuda_info()
    print(f"✅ mindruntime backend: {info}")

    eng = BrainEngineV2(width=32, height=32)
    frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    for _ in range(3):
        eng.step(frame)
    stats = eng.get_statistics()
    print(f"✅ BrainEngineV2 step OK — coherence_mean={stats['coherence_mean']:.4f}")
    print("\n🔥 Setup OK! Pronto per: python -m mindruntime.visualizer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
