"""Template risonatori — forme in Fourier 2D per riconoscimento per interferenza."""

from __future__ import annotations

import math
from typing import Any

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TEMPLATE_SIZE = 16


def _letter_seed(ch: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(ch))


def fourier_template(ch: str, size: int = TEMPLATE_SIZE) -> list[list[float]]:
    """Pattern di fase/frequenza per un simbolo — zero training."""
    if not HAS_NUMPY or np is None:
        return [[0.0] * size for _ in range(size)]
    seed = _letter_seed(ch)
    rng = np.random.RandomState(seed % 2**31)
    out = np.zeros((size, size), dtype=np.float32)
    cx, cy = size / 2, size / 2
    for ky in range(4):
        for kx in range(4):
            freq = 0.4 + (kx + ky) * 0.35 + (seed % 7) * 0.05
            ang = rng.uniform(0, 2 * math.pi)
            for y in range(size):
                for x in range(size):
                    dx, dy = x - cx, y - cy
                    out[y, x] += math.cos(freq * dx + ang) * math.cos(freq * dy + ang * 0.7)
    out -= out.min()
    mx = out.max()
    if mx > 0:
        out /= mx
    return out.tolist()


def build_template_bank() -> dict[str, Any]:
    bank = {ch: fourier_template(ch) for ch in SYMBOLS}
    return {"symbols": list(SYMBOLS), "templates": bank, "size": TEMPLATE_SIZE}


def correlate_template(
    patch: list[list[float]],
    template: list[list[float]],
) -> float:
    if not patch or not template or not HAS_NUMPY or np is None:
        return 0.0
    a = np.array(patch, dtype=np.float32)
    b = np.array(template, dtype=np.float32)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h < 2 or w < 2:
        return 0.0
    a = a[:h, :w].ravel()
    b = b[:h, :w].ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
