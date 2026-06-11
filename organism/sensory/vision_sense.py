"""Vision sense — feature gerarchiche, motion skip, processing async-friendly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from organism.sensory.object_scene import feature_similarity
from organism.sensory.visual_scene import scene_features, scene_signature


@dataclass
class HierarchicalFeatures:
    edges: float = 0.0
    shapes: list[str] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)
    whole: str = ""
    motion: float = 0.0
    gist: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": round(self.edges, 4),
            "shapes": self.shapes,
            "parts": self.parts,
            "whole": self.whole,
            "motion": round(self.motion, 4),
            "gist": self.gist,
        }


class VisionSense:
    """Pipeline visiva — edge → shape → parts → whole + optical flow leggero."""

    MOTION_SKIP_THRESHOLD = 0.05

    def __init__(self) -> None:
        self._prev_grid: list[list[int]] | None = None
        self._prev_sig = ""
        self._skip_count = 0

    def should_skip_frame(self, grid: list[list[int]]) -> bool:
        """Skip frame se diff < 5% — riduce latenza."""
        if not grid or not self._prev_grid:
            return False
        diff = self._frame_diff(grid, self._prev_grid)
        return diff < self.MOTION_SKIP_THRESHOLD

    def process(
        self,
        grid: list[list[int]],
        *,
        rgb_grid: list[list[tuple[int, int, int]]] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        if not grid or not grid[0]:
            return {"skipped": True, "features": {}, "sig": "empty"}

        if not force and self.should_skip_frame(grid):
            self._skip_count += 1
            return {
                "skipped": True,
                "sig": self._prev_sig,
                "features": self._last_features(),
                "skip_count": self._skip_count,
            }

        sig = scene_signature(grid)
        base = scene_features(grid)
        hier = self._hierarchical(grid, rgb_grid)
        motion = self._optical_flow(grid)

        features = {
            **base,
            "object_sig": sig,
            "edge_density": hier.edges,
            "shapes": hier.shapes,
            "parts": hier.parts,
            "whole_hint": hier.whole,
            "motion": motion,
            "hierarchical": hier.to_dict(),
            "gist": hier.gist,
        }

        self._prev_grid = [row[:] for row in grid]
        self._prev_sig = sig
        self._last = features
        return {"skipped": False, "sig": sig, "features": features, "motion": motion}

    def _last_features(self) -> dict[str, Any]:
        return getattr(self, "_last", {})

    def _frame_diff(self, a: list[list[int]], b: list[list[int]]) -> float:
        h = min(len(a), len(b))
        w = min(len(a[0]) if a else 0, len(b[0]) if b else 0)
        if h < 2 or w < 2:
            return 1.0
        total = 0.0
        n = 0
        step_y = max(1, h // 16)
        step_x = max(1, w // 16)
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                total += abs(a[y][x] - b[y][x]) / 255.0
                n += 1
        return total / max(1, n)

    def _optical_flow(self, grid: list[list[int]]) -> float:
        if self._prev_grid is None:
            return 0.0
        return self._frame_diff(grid, self._prev_grid)

    def _hierarchical(
        self,
        grid: list[list[int]],
        rgb_grid: list[list[tuple[int, int, int]]] | None,
    ) -> HierarchicalFeatures:
        h = len(grid)
        w = len(grid[0]) if grid else 0
        edges = 0.0
        shapes: list[str] = []
        parts: list[str] = []

        # Edge layer — gradiente semplice
        edge_count = 0
        samples = 0
        for y in range(1, h - 1, max(1, h // 12)):
            for x in range(1, w - 1, max(1, w // 12)):
                gx = abs(int(grid[y][x + 1]) - int(grid[y][x - 1]))
                gy = abs(int(grid[y + 1][x]) - int(grid[y - 1][x]))
                if gx + gy > 40:
                    edge_count += 1
                samples += 1
        edges = edge_count / max(1, samples)

        # Shape primitives
        aspect = w / max(1, h)
        if 0.85 < aspect < 1.15:
            shapes.append("quadrato")
        elif aspect > 1.4:
            shapes.append("allungato")
        else:
            shapes.append("ovale")

        lum = sum(grid[y][x] for y in range(0, h, max(1, h // 8)) for x in range(0, w, max(1, w // 8)))
        lum /= max(1, (h // max(1, h // 8)) * (w // max(1, w // 8)))
        if lum < 60:
            parts.append("ombra")
        elif lum > 200:
            parts.append("luce")
        if edges > 0.25:
            parts.append("contorno")

        color = ""
        if rgb_grid:
            rs = gs = bs = n = 0
            for y in range(0, h, max(1, h // 8)):
                for x in range(0, w, max(1, w // 8)):
                    r, g, b = rgb_grid[y][x]
                    rs += r
                    gs += g
                    bs += b
                    n += 1
            if n:
                color = _dominant_color(rs / n, gs / n, bs / n)

        gist = {
            "edge_density": edges,
            "aspect_ratio": aspect,
            "luminance": lum / 255.0,
            "color": color,
        }
        whole = shapes[0] if shapes else "forma"
        return HierarchicalFeatures(edges=edges, shapes=shapes, parts=parts, whole=whole, gist=gist)


def _dominant_color(r: float, g: float, b: float) -> str:
    if r > g + 30 and r > b + 30:
        return "rosso"
    if g > r + 20 and g > b + 20:
        return "verde"
    if b > r + 20 and b > g + 20:
        return "blu"
    if r > 200 and g > 200 and b > 200:
        return "bianco"
    if r < 50 and g < 50 and b < 50:
        return "nero"
    if r > 180 and g > 100 and b < 80:
        return "giallo"
    return "grigio"


def compare_prototypes(a: dict[str, Any], b: dict[str, Any]) -> float:
    return feature_similarity(a, b)
