"""Template risonatori — forme geometriche + lettere, FFT 2D pre-calcolata."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.fft import fft2, ifft2

    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False

DEFAULT_SIZE = 32
TEMPLATE_NAMES = (
    "circle",
    "square",
    "triangle",
    "hline",
    "vline",
    "diag",
    "A",
    "B",
    "C",
    "D",
    "E",
)


def _blank(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=np.float32)


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr -= arr.min()
    mx = arr.max()
    if mx > 1e-9:
        arr /= mx
    return arr


def shape_circle(size: int = DEFAULT_SIZE, *, radius: float | None = None) -> np.ndarray:
    r = radius if radius is not None else size * 0.32
    cx, cy = size / 2, size / 2
    out = _blank(size)
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                out[y, x] = 1.0
    return out


def shape_square(size: int = DEFAULT_SIZE) -> np.ndarray:
    out = _blank(size)
    m = size // 4
    out[m : size - m, m : size - m] = 1.0
    return out


def shape_triangle(size: int = DEFAULT_SIZE) -> np.ndarray:
    out = _blank(size)
    apex = size // 6
    base = size - size // 6
    for y in range(apex, base):
        span = int((y - apex) / max(1, base - apex) * (size // 2))
        cx = size // 2
        out[y, cx - span : cx + span] = 1.0
    return out


def shape_hline(size: int = DEFAULT_SIZE) -> np.ndarray:
    out = _blank(size)
    out[size // 2, size // 6 : size - size // 6] = 1.0
    return out


def shape_vline(size: int = DEFAULT_SIZE) -> np.ndarray:
    out = _blank(size)
    out[size // 6 : size - size // 6, size // 2] = 1.0
    return out


def shape_diag(size: int = DEFAULT_SIZE) -> np.ndarray:
    out = _blank(size)
    for i in range(size // 6, size - size // 6):
        out[i, i] = 1.0
        if i + 1 < size:
            out[i, i + 1] = 0.6
    return out


def _letter_bitmap(ch: str, size: int) -> np.ndarray:
    """Pattern semplificato per lettere A–E (senza font esterno)."""
    out = _blank(size)
    s = size
    if ch == "A":
        for y in range(s // 4, s - s // 6):
            t = (y - s // 4) / max(1, s - s // 4 - s // 6)
            half = int((1 - t) * s * 0.22 + s * 0.06)
            cx = s // 2
            out[y, cx - half : cx + half] = 1.0
        out[s // 2, s // 4 : s - s // 4] = 0.85
    elif ch == "B":
        out[s // 5 : s - s // 5, s // 4 : s // 4 + 2] = 1.0
        out[s // 5 : s // 2, s // 4 : s - s // 4] = 1.0
        out[s // 2 : s - s // 5, s // 4 : s - s // 4] = 1.0
    elif ch == "C":
        for y in range(s // 5, s - s // 5):
            out[y, s // 4] = 1.0
            if y < s // 2:
                out[y, s // 4 : s - s // 3] = 0.5
            else:
                out[y, s // 4 : s - s // 3] = 0.5
    elif ch == "D":
        out[s // 5 : s - s // 5, s // 4 : s // 4 + 2] = 1.0
        for y in range(s // 5, s - s // 5):
            t = abs(y - s // 2) / (s // 2)
            x = int(s // 4 + (1 - t) * s * 0.35)
            out[y, x : x + 2] = 1.0
    elif ch == "E":
        out[s // 5 : s - s // 5, s // 4 : s // 4 + 2] = 1.0
        out[s // 5, s // 4 : s - s // 4] = 1.0
        out[s // 2, s // 4 : s - s // 3] = 1.0
        out[s - s // 5, s // 4 : s - s // 4] = 1.0
    return out


def spatial_template(name: str, size: int = DEFAULT_SIZE) -> np.ndarray:
    builders = {
        "circle": shape_circle,
        "square": shape_square,
        "triangle": shape_triangle,
        "hline": shape_hline,
        "vline": shape_vline,
        "diag": shape_diag,
    }
    if name in builders:
        return _normalize(builders[name](size))
    if name in "ABCDE":
        return _normalize(_letter_bitmap(name, size))
    raise ValueError(f"template sconosciuto: {name}")


def fft_template(name: str, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Template nel dominio delle frequenze (modulo FFT)."""
    spatial = spatial_template(name, size)
    if HAS_SCIPY:
        spec = fft2(spatial)
        mag = np.abs(spec).astype(np.float32)
        return _normalize(mag)
    # fallback senza scipy: DCT approssimata via cosines
    out = _blank(size)
    for ky in range(4):
        for kx in range(4):
            for y in range(size):
                for x in range(size):
                    out[y, x] += spatial[y, x] * math.cos(kx * x / size * math.pi) * math.cos(
                        ky * y / size * math.pi
                    )
    return _normalize(out)


def build_resonator_bank(
    names: tuple[str, ...] = TEMPLATE_NAMES,
    size: int = DEFAULT_SIZE,
    *,
    use_fft: bool = False,
) -> dict[str, Any]:
    """Banca template per match_resonators."""
    spatial: dict[str, np.ndarray] = {}
    spectral: dict[str, np.ndarray] = {}
    for name in names:
        spatial[name] = spatial_template(name, size)
        spectral[name] = fft_template(name, size) if use_fft else spatial[name]
    stack = np.stack([spatial[n] for n in names], axis=0)
    return {
        "names": list(names),
        "size": size,
        "spatial": spatial,
        "spectral": spectral,
        "stack": stack,
    }
