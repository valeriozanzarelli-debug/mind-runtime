# CUDA optimization — Mindruntime GPU V2

## What changed

| Kernel | Optimization |
|--------|----------------|
| **Turing RD** | 18×18 shared-memory tile (halo 1), impulse+phase channels |
| **Gamma binding** | 22×22 shared tile (radius 3), phase reads from shared mem |
| **SOC avalanche** | 26×26 shared tile (radius 5), impulse+voltage |
| **Predictive coding** | 18×18 shared tile, impulse+weight |
| **Inject RGB** | New CUDA kernel (was CPU-only) |
| **Hodgkin-Huxley** | Grid-stride unchanged; batched sync with other kernels |
| **Init field** | Unchanged (called once at birth) |

## Host-side

- `ORGANISM_HH_SUBSTEPS` (default `1`) — was hardcoded `2` per tick
- Single `cuda.synchronize()` at end of `physics_step_v2` when optimized path active
- CPU fallback unchanged for CI / no-GPU

## Files

- `mindruntime/gpu_physics_v2_cuda_optimized.py` — tiled kernels
- `mindruntime/gpu_physics_v2.py` — dispatches to optimized or legacy CUDA / CPU
- `scripts/benchmark_fps.py` — measure sustained FPS

## Benchmark

```bash
pip install -e ".[mindruntime]"
python scripts/benchmark_fps.py --width 256 --height 256 --steps 100
```

Target: **≥30 FPS** on RTX 1060 @ 256×256 with `ORGANISM_HH_SUBSTEPS=1`.

## Notes

- Phase-gradient is folded into Turing RD (no separate kernel).
- Free-energy post-pass still uses CPU `np.roll` — minor vs stencil cost.
- Channel-first layout `(12,H,W)` not implemented; revisit if FPS still below target.
