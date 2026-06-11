"""Similarità feature oggetto — per riconoscimento gerarchico."""

from __future__ import annotations

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
    _str("color", 2.0)
    _str("scene_type", 1.0)
    _str("object_sig", 2.5)

    gist_a = a.get("gist") or {}
    gist_b = b.get("gist") or {}
    if gist_a and gist_b:
        score += feature_similarity(gist_a, gist_b) * 1.5
        weights += 1.5

    return score / max(1.0, weights)


def analyze_objects(features: dict[str, Any], *, limit: int = 6) -> list[str]:
    """Estrae etichette oggetto da feature scena — compat server."""
    out: list[str] = []
    whole = str(features.get("whole_hint") or features.get("whole") or "")
    if whole:
        out.append(whole)
    for p in features.get("parts", []) or []:
        if p and p not in out:
            out.append(str(p))
    for s in features.get("shapes", []) or []:
        if s and s not in out:
            out.append(str(s))
    color = str(features.get("color", ""))
    if color and color not in out:
        out.insert(0, color)
    return out[:limit]
