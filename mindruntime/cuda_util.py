"""Rilevamento CUDA / Numba — RTX 1060 e fallback CPU."""

from __future__ import annotations

HAS_NUMBA = False
HAS_CUDA = False

try:
    from numba import cuda as numba_cuda

    HAS_NUMBA = True
    HAS_CUDA = bool(numba_cuda.is_available())
except ImportError:  # pragma: no cover
    numba_cuda = None  # type: ignore


def cuda_info() -> dict[str, object]:
    if not HAS_NUMBA or numba_cuda is None:
        return {"numba": False, "cuda": False, "device": "cpu"}
    if not HAS_CUDA:
        return {"numba": True, "cuda": False, "device": "cpu"}
    try:
        dev = numba_cuda.get_current_device()
        return {
            "numba": True,
            "cuda": True,
            "device": dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name),
            "compute_capability": dev.compute_capability,
        }
    except Exception as exc:  # pragma: no cover
        return {"numba": True, "cuda": False, "device": "cpu", "error": str(exc)}
