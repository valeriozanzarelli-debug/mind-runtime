"""Percorso visivo per volti — forma, pelle, occhi, bocca (senza CNN)."""

from __future__ import annotations

import math
from typing import Any


def _skin_mask(rgb: list[list[tuple[int, int, int]]]) -> list[list[bool]]:
    h = len(rgb)
    w = len(rgb[0]) if h else 0
    mask = [[False] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r, g, b = rgb[y][x]
            if r < 60 or g < 30:
                continue
            if r > g > b and (r - g) < 90 and (g - b) > 5:
                mask[y][x] = True
            elif r > 95 and g > 50 and b > 30 and r >= g and g >= b * 0.85:
                mask[y][x] = True
    return mask


def _region_stats(
    gray: list[list[int]],
    rgb: list[list[tuple[int, int, int]]] | None,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> dict[str, float]:
    h = len(gray)
    w = len(gray[0]) if h else 0
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y1 <= y0 or x1 <= x0:
        return {"mean_lum": 0.0, "edge": 0.0, "dark_ratio": 0.0}
    lums: list[float] = []
    edges = 0.0
    dark = 0
    n = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            lum = gray[y][x] / 255.0
            lums.append(lum)
            if lum < 0.28:
                dark += 1
            if 0 < y < h - 1 and 0 < x < w - 1:
                gx = abs(gray[y][x + 1] - gray[y][x - 1])
                gy = abs(gray[y + 1][x] - gray[y - 1][x])
                edges += math.hypot(gx, gy) / 255.0
            n += 1
    return {
        "mean_lum": sum(lums) / max(1, len(lums)),
        "edge": edges / max(1, n),
        "dark_ratio": dark / max(1, n),
    }


def analyze_face(
    grid: list[list[int]],
    *,
    rgb_grid: list[list[tuple[int, int, int]]] | None = None,
) -> dict[str, Any]:
    """Estrae indizi facciali — volto generico + gist per binding."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    if not w or not h:
        return _empty()

    rgb = rgb_grid
    if not rgb:
        rgb = [[(grid[y][x], grid[y][x], grid[y][x]) for x in range(w)] for y in range(h)]

    mask = _skin_mask(rgb)
    skin_pixels = sum(1 for row in mask for c in row if c)
    skin_ratio = skin_pixels / max(1, w * h)

    # bbox pelle
    ys = [y for y in range(h) for x in range(w) if mask[y][x]]
    xs = [x for y in range(h) for x in range(w) if mask[y][x]]
    if len(xs) < max(12, (w * h) // 80):
        return _empty_partial(skin_ratio)

    y_min, y_max = min(ys), max(ys)
    x_min, x_max = min(xs), max(xs)
    fh = y_max - y_min + 1
    fw = x_max - x_min + 1
    aspect = fw / max(1, fh)
    cx = (x_min + x_max) / 2 / w
    cy = (y_min + y_max) / 2 / h
    centered = 1.0 - min(1.0, math.hypot(cx - 0.5, cy - 0.42) * 2.2)

    eye_y0 = y_min + int(fh * 0.18)
    eye_y1 = y_min + int(fh * 0.48)
    mouth_y0 = y_min + int(fh * 0.58)
    mouth_y1 = y_min + int(fh * 0.88)
    eye_l = _region_stats(grid, rgb, eye_y0, eye_y1, x_min, x_min + fw // 2)
    eye_r = _region_stats(grid, rgb, eye_y0, eye_y1, x_min + fw // 2, x_max + 1)
    mouth = _region_stats(grid, rgb, mouth_y0, mouth_y1, x_min, x_max + 1)
    brow = _region_stats(grid, rgb, y_min, eye_y0, x_min, x_max + 1)

    eye_contrast = (eye_l["dark_ratio"] + eye_r["dark_ratio"]) / 2
    mouth_curve = mouth["mean_lum"] - (eye_l["mean_lum"] + eye_r["mean_lum"]) / 2
    smile_hint = max(0.0, mouth_curve * 1.8 + mouth["edge"] * 0.4)
    frown_hint = max(0.0, brow["edge"] * 0.5 - mouth_curve)
    surprise_hint = max(0.0, eye_contrast * 0.6 + brow["edge"] * 0.3)

    oval_ok = 0.65 <= aspect <= 1.45 and fh >= h * 0.22
    score = min(
        1.0,
        skin_ratio * 2.2
        + (0.25 if oval_ok else 0.0)
        + centered * 0.25
        + min(0.35, eye_contrast * 1.2),
    )
    detected = score >= 0.42 and oval_ok

    gist = {
        "skin_ratio": round(skin_ratio, 4),
        "face_aspect": round(aspect, 3),
        "eye_contrast": round(eye_contrast, 4),
        "mouth_curve": round(mouth_curve, 4),
        "smile_hint": round(smile_hint, 4),
        "frown_hint": round(frown_hint, 4),
        "surprise_hint": round(surprise_hint, 4),
        "brow_edge": round(brow["edge"], 4),
        "centered": round(centered, 3),
        "face_score": round(score, 3),
    }

    symbols: list[str] = []
    if detected:
        symbols.append("FACE:detected")
        if smile_hint > 0.12:
            symbols.append("FACE:smile")
        if frown_hint > 0.1:
            symbols.append("FACE:tension")

    return {
        "detected": detected,
        "face_score": round(score, 3),
        "face_sig": _face_sig(gist),
        "gist": gist,
        "symbols": symbols,
        "bbox": {"x0": x_min, "y0": y_min, "x1": x_max, "y1": y_max},
    }


def _face_sig(gist: dict[str, float]) -> str:
    parts = [
        f"s{int(gist.get('skin_ratio', 0) * 100)}",
        f"a{int(gist.get('face_aspect', 1) * 10)}",
        f"e{int(gist.get('eye_contrast', 0) * 100)}",
        f"m{int((gist.get('mouth_curve', 0) + 0.2) * 50)}",
    ]
    return "face_" + "_".join(parts)


def _empty() -> dict[str, Any]:
    return {
        "detected": False,
        "face_score": 0.0,
        "face_sig": "",
        "gist": {},
        "symbols": [],
        "bbox": {},
    }


def _empty_partial(skin_ratio: float) -> dict[str, Any]:
    return {
        "detected": False,
        "face_score": round(min(0.35, skin_ratio * 1.5), 3),
        "face_sig": "",
        "gist": {"skin_ratio": round(skin_ratio, 4)},
        "symbols": [],
        "bbox": {},
    }


def face_gist_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    keys = (
        "skin_ratio", "face_aspect", "eye_contrast", "mouth_curve",
        "smile_hint", "frown_hint", "surprise_hint", "brow_edge",
    )
    ga = a if "skin_ratio" in a else a.get("gist", {})
    gb = b if "skin_ratio" in b else b.get("gist", {})
    if not ga or not gb:
        return 0.0
    dist = 0.0
    for k in keys:
        va = float(ga.get(k, 0))
        vb = float(gb.get(k, 0))
        scale = 0.35 if k == "face_aspect" else 0.18
        dist += abs(va - vb) / scale
    return max(0.0, 1.0 - dist / len(keys))
