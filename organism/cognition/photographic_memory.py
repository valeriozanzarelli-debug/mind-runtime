"""Memoria fotografica — istantanee visive con gist e etichette."""

from __future__ import annotations

import hashlib
import time
from typing import Any


class PhotographicMemory:
    """Buffer ad alta capacità di scene viste — richiamo per similarità gist."""

    def __init__(self, *, capacity: int = 2000) -> None:
        self.capacity = max(100, capacity)
        self._frames: list[dict[str, Any]] = []

    def capture(
        self,
        *,
        object_sig: str,
        features: dict[str, Any],
        labels: list[str],
        thumb_gray: list[int] | None = None,
    ) -> str:
        gist = {
            "object_sig": object_sig,
            "color": features.get("color", ""),
            "scene_type": features.get("scene_type", ""),
            "luminance": features.get("luminance", 0),
            "face": (features.get("face") or {}).get("face_score", 0),
        }
        key = hashlib.sha256(str(gist).encode()).hexdigest()[:16]
        entry = {
            "key": key,
            "t": time.time(),
            "gist": gist,
            "labels": labels[:8],
            "thumb": (thumb_gray or [])[:256],
        }
        self._frames.append(entry)
        if len(self._frames) > self.capacity:
            self._frames = self._frames[-self.capacity :]
        return key

    def recall(self, features: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
        osig = str(features.get("object_sig", ""))
        color = str(features.get("color", ""))
        scored: list[tuple[float, dict[str, Any]]] = []
        for fr in self._frames[-500:]:
            g = fr.get("gist", {})
            score = 0.0
            if osig and g.get("object_sig") == osig:
                score += 3.0
            if color and g.get("color") == color:
                score += 1.5
            if score > 0:
                scored.append((score, fr))
        scored.sort(key=lambda x: -x[0])
        return [fr for _, fr in scored[:limit]]

    def label_hints(self, features: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for fr in self.recall(features, limit=3):
            for lb in fr.get("labels", []):
                if lb and lb not in out:
                    out.append(lb)
        return out[:6]

    def stats(self) -> dict[str, Any]:
        return {"frames": len(self._frames), "capacity": self.capacity}

    def to_dict(self) -> dict[str, Any]:
        return {"frames": self._frames[-min(400, self.capacity) :], "capacity": self.capacity}

    def load_dict(self, data: dict[str, Any]) -> None:
        self.capacity = int(data.get("capacity", self.capacity))
        self._frames = list(data.get("frames", []))[-self.capacity :]
