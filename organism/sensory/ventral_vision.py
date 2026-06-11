"""Ventral stream — wrapper su VisionSense per compatibilità baby_agent."""

from __future__ import annotations

from typing import Any

from organism.sensory.vision_sense import VisionSense

_sense = VisionSense()


def process_ventral_stream(
    grid: list[list[int]],
    *,
    rgb_grid: list[list[tuple[int, int, int]]] | None = None,
    color_rgb: dict[str, float] | None = None,
) -> dict[str, Any]:
    result = _sense.process(grid, rgb_grid=rgb_grid)
    feat = dict(result.get("features", {}))
    if color_rgb:
        feat["color_rgb"] = color_rgb
    symbols: list[str] = []
    color = str(feat.get("color", "") or feat.get("gist", {}).get("color", ""))
    if color:
        symbols.append(f"COL:{color}")
    shapes = feat.get("shapes") or []
    for sh in shapes[:2]:
        symbols.append(f"SHAPE:{sh}")
    if feat.get("motion", 0) > 0.08:
        symbols.append("VIS:motion")
    return {
        "features": feat,
        "symbols": symbols,
        "object_sig": feat.get("object_sig", ""),
        "skipped": result.get("skipped", False),
    }
