"""Object analysis from grayscale grids — blob detection and shape classification."""

from __future__ import annotations

import hashlib
from typing import Any


def feature_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """0..1 — confronto gist visivo."""
    if not a or not b:
        return 0.0
    score = 0.0
    weights = 0.0

    def _num(key: str, w: float = 1.0) -> None:
        nonlocal score, weights
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            return
        try:
            fa, fb = float(va), float(vb)
            diff = abs(fa - fb)
            score += w * max(0.0, 1.0 - diff)
            weights += w
        except (TypeError, ValueError):
            pass

    def _str(key: str, w: float = 1.0) -> None:
        nonlocal score, weights
        sa, sb = str(a.get(key, "")), str(b.get(key, ""))
        if sa and sb:
            score += w * (1.0 if sa == sb else 0.0)
            weights += w

    _num("luminance", 1.2)
    _num("contrast", 1.0)
    _num("edge_density", 1.5)
    _num("aspect_ratio", 0.8)
    _num("compactness", 1.2)
    _str("color", 2.0)
    _str("scene_type", 1.0)
    _str("object_sig", 2.5)

    gist_a = a.get("gist") or {}
    gist_b = b.get("gist") or {}
    if gist_a and gist_b:
        score += feature_similarity(gist_a, gist_b) * 1.5
        weights += 1.5

    return score / max(1.0, weights)


def _find_blobs(grid: list[list[int]], threshold: int) -> list[list[tuple[int, int]]]:
    """Flood-fill BFS to find connected bright regions."""
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    blobs: list[list[tuple[int, int]]] = []
    for sy in range(h):
        for sx in range(w):
            if grid[sy][sx] > threshold and not visited[sy][sx]:
                blob: list[tuple[int, int]] = []
                queue = [(sy, sx)]
                visited[sy][sx] = True
                while queue:
                    cy, cx = queue.pop()
                    blob.append((cy, cx))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny][nx] and grid[ny][nx] > threshold:
                            visited[ny][nx] = True
                            queue.append((ny, nx))
                blobs.append(blob)
    return blobs


def _blob_fill_ratio(blob: list[tuple[int, int]]) -> float:
    """Area / bounding-box area — near 1.0 for squares, ~0.78 for circles."""
    if not blob:
        return 0.0
    ys = [p[0] for p in blob]
    xs = [p[1] for p in blob]
    h = max(ys) - min(ys) + 1
    w = max(xs) - min(xs) + 1
    return len(blob) / max(1, h * w)


def analyze_objects(grid: list[list[int]], *, limit: int = 6) -> dict[str, Any]:
    """
    Analyses a grayscale pixel grid and returns structured object features.

    Returns a dict with:
        blob_count  – number of bright blobs detected
        symbols     – shape labels (e.g. "rotondo", "quadrato")
        features    – numeric feature dict for VisualBinder
        object_sig  – compact scene hash
    """
    from organism.sensory.visual_scene import scene_features, scene_signature

    feat = scene_features(grid)
    sig = scene_signature(grid)

    if not grid or not grid[0]:
        return {"blob_count": 0, "symbols": [], "features": feat, "object_sig": sig}

    pixels = [grid[y][x] for y in range(len(grid)) for x in range(len(grid[0]))]
    mean = sum(pixels) / max(1, len(pixels))
    threshold = int(mean + (max(pixels) - mean) * 0.35)
    threshold = max(threshold, 50)

    blobs = _find_blobs(grid, threshold)
    blobs = [b for b in blobs if len(b) >= 4]

    symbols: list[str] = []
    fill_vals: list[float] = []
    for blob in blobs:
        f = _blob_fill_ratio(blob)
        fill_vals.append(f)
        # Circles fill ~78% of their bounding box; squares fill ~100%.
        if f < 0.87:
            symbols.append("rotondo")
        else:
            symbols.append("quadrato")

    avg_fill = sum(fill_vals) / max(1, len(fill_vals))

    flat_feat: dict[str, Any] = {
        "luminance": feat.get("luminance", 0.0),
        "contrast": feat.get("contrast", 0.0),
        "edge_density": feat.get("contrast", 0.0),
        "fill_ratio": round(avg_fill, 4),
        "blob_count": len(blobs),
        "aspect_ratio": 1.0,
        "scene_type": feat.get("scene_type", "oggetto"),
        "object_sig": sig,
    }

    return {
        "blob_count": len(blobs),
        "symbols": list(dict.fromkeys(symbols))[:limit],
        "features": flat_feat,
        "object_sig": sig,
    }
