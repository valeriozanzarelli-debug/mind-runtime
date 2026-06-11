"""Feature scena da griglia grayscale — compat layer esistente."""

from __future__ import annotations

import hashlib
from typing import Any


def scene_signature(grid: list[list[int]]) -> str:
    if not grid or not grid[0]:
        return "empty"
    h, w = len(grid), len(grid[0])
    sample = []
    for y in range(0, h, max(1, h // 8)):
        for x in range(0, w, max(1, w // 8)):
            sample.append(grid[y][x])
    return hashlib.sha256(bytes(sample[:64])).hexdigest()[:16]


def scene_features(grid: list[list[int]]) -> dict[str, Any]:
    if not grid or not grid[0]:
        return {"luminance": 0.0, "contrast": 0.0, "scene_type": "vuoto"}
    h, w = len(grid), len(grid[0])
    pixels = [grid[y][x] for y in range(h) for x in range(w)]
    mean = sum(pixels) / max(1, len(pixels))
    var = sum((p - mean) ** 2 for p in pixels) / max(1, len(pixels))
    lum = mean / 255.0
    contrast = min(1.0, (var ** 0.5) / 128.0)
    scene_type = "oggetto"
    if lum < 0.2:
        scene_type = "buio"
    elif lum > 0.75 and contrast < 0.15:
        scene_type = "distesa"
    elif contrast > 0.4:
        scene_type = "dettaglio"
    return {
        "luminance": round(lum, 4),
        "contrast": round(contrast, 4),
        "scene_type": scene_type,
        "width": w,
        "height": h,
    }
