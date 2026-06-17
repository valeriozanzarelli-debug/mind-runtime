"""Template risonatori — forme geometriche + lettere A-Z, FFT 2D pre-calcolata."""

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
    """Pattern geometrico semplificato per lettere."""
    out = _blank(size)
    s = size
    cx = s // 2
    if ch == "A":
        for y in range(s // 4, s - s // 6):
            t = (y - s // 4) / max(1, s - s // 4 - s // 6)
            half = int((1 - t) * s * 0.22 + s * 0.06)
            out[y, cx - half : cx + half] = 1.0
        out[s // 2, s // 4 : s - s // 4] = 0.85
    elif ch == "B":
        out[s // 5 : s - s // 5, s // 4 : s // 4 + 2] = 1.0
        out[s // 5 : s // 2, s // 4 : s - s // 4] = 1.0
        out[s // 2 : s - s // 5, s // 4 : s - s // 4] = 1.0
    elif ch == "C":
        for y in range(s // 5, s - s // 5):
            out[y, s // 4] = 1.0
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
    elif ch == "F":
        out[s // 5 : s - s // 5, s // 4 : s // 4 + 2] = 1.0
        out[s // 5, s // 4 : s - s // 4] = 1.0
        out[s // 2, s // 4 : s - s // 3] = 1.0
    elif ch == "H":
        out[s // 5 : s - s // 5, s // 4 : s // 4 + 2] = 1.0
        out[s // 5 : s - s // 5, s - s // 4 : s - s // 4 + 2] = 1.0
        out[s // 2, s // 4 : s - s // 4] = 1.0
    elif ch == "I":
        out[s // 5 : s - s // 5, cx : cx + 2] = 1.0
        out[s // 5, s // 4 : s - s // 4] = 1.0
        out[s - s // 5, s // 4 : s - s // 4] = 1.0
    elif ch == "L":
        out[s // 5 : s - s // 5, s // 4 : s // 4 + 2] = 1.0
        out[s - s // 5, s // 4 : s - s // 4] = 1.0
    elif ch == "O":
        for y in range(s // 5, s - s // 5):
            for x in range(s // 5, s - s // 5):
                if abs((x - cx) ** 2 + (y - s // 2) ** 2 - (s * 0.22) ** 2) < s * 0.8:
                    out[y, x] = 1.0
    elif ch == "T":
        out[s // 5, s // 4 : s - s // 4] = 1.0
        out[s // 5 : s - s // 5, cx : cx + 2] = 1.0
    elif ch == "X":
        for i in range(s // 5, s - s // 5):
            out[i, i] = 1.0
            out[i, s - 1 - i] = 1.0
    elif ch == "Z":
        out[s // 5, s // 4 : s - s // 4] = 1.0
        for i in range(s // 5, s - s // 5):
            out[i, s - 1 - i] = 1.0
        out[s - s // 5, s // 4 : s - s // 4] = 1.0
    else:
        # pattern a barre verticali distinte per lettera (placeholder geometrico)
        idx = ord(ch) - ord("A")
        bar = s // 5 + (idx % 5) * 2
        if bar < s - s // 5:
            out[s // 5 : s - s // 5, bar : bar + 2] = 1.0
        out[s // 5, s // 4 : s - s // 4] = 0.4
        out[s - s // 5, s // 4 : s - s // 4] = 0.4
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
    if len(name) == 1 and name.isalpha():
        return _normalize(_letter_bitmap(name.upper(), size))
    raise ValueError(f"template sconosciuto: {name}")


def fft_template(name: str, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Template nel dominio delle frequenze."""
    spatial = spatial_template(name, size)
    if HAS_SCIPY:
        return fft2(spatial)
    out = _blank(size)
    for ky in range(4):
        for kx in range(4):
            for y in range(size):
                for x in range(size):
                    out[y, x] += spatial[y, x] * math.cos(kx * x / size * math.pi) * math.cos(
                        ky * y / size * math.pi
                    )
    return _normalize(out)


def create_resonator_circle(size: int = DEFAULT_SIZE, radius: float | None = None) -> dict[str, Any]:
    pattern = shape_circle(size, radius=radius)
    return {"name": "circle", "pattern": pattern, "fft": fft_template("circle", size), "symbol": "●"}


def create_resonator_square(size: int = DEFAULT_SIZE) -> dict[str, Any]:
    pattern = shape_square(size)
    return {"name": "square", "pattern": pattern, "fft": fft_template("square", size), "symbol": "■"}


def create_resonator_letters(size: int = DEFAULT_SIZE) -> list[dict[str, Any]]:
    """26 template per lettere A-Z."""
    letters = []
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        pattern = spatial_template(char, size)
        letters.append({
            "name": f"letter_{char}",
            "pattern": pattern,
            "fft": fft2(pattern) if HAS_SCIPY else pattern,
            "symbol": char,
        })
    return letters


def load_resonators_from_disk(size: int = DEFAULT_SIZE) -> list[dict[str, Any]]:
    """Carica banca risonatori pre-calcolati (in-memory, no I/O disco)."""
    resonators: list[dict[str, Any]] = []
    resonators.append(create_resonator_circle(size))
    resonators.append(create_resonator_square(size))
    resonators.extend(create_resonator_letters(size))
    return resonators


def correlate_2d(field: np.ndarray, resonator: dict[str, Any]) -> float:
    """Correlazione normalizzata campo vs template risonatore."""
    pattern = resonator.get("pattern")
    if pattern is None:
        return 0.0
    tpl = pattern.astype(np.float32)
    th, tw = tpl.shape
    h, w = field.shape[:2]
    y0, x0 = max(0, (h - th) // 2), max(0, (w - tw) // 2)
    patch = field[y0 : y0 + th, x0 : x0 + tw]
    if patch.shape != tpl.shape:
        return 0.0
    if HAS_SCIPY and "fft" in resonator:
        f_field = fft2(patch)
        corr = ifft2(f_field * np.conj(resonator["fft"]))
        return float(np.max(np.abs(corr)) / (np.linalg.norm(patch) * np.linalg.norm(tpl) + 1e-9))
    dot = float(np.sum(patch * tpl))
    na = float(np.sum(patch * patch)) + 1e-9
    nb = float(np.sum(tpl * tpl)) + 1e-9
    return dot / math.sqrt(na * nb)


def correlate_resonators(
    field: np.ndarray,
    resonators: list[dict[str, Any]],
) -> list[tuple[str, float]]:
    """Correla campo con tutti i risonatori; ritorna lista ordinata per confidenza."""
    scores: list[tuple[str, float]] = []
    for res in resonators:
        conf = correlate_2d(field, res)
        scores.append((str(res["symbol"]), conf))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def build_resonator_bank(
    names: tuple[str, ...] = TEMPLATE_NAMES,
    size: int = DEFAULT_SIZE,
    *,
    use_fft: bool = False,
) -> dict[str, Any]:
    """Banca template per match legacy (gpu_core / dendritic)."""
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
