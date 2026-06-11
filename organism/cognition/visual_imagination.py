"""Visual imagination — flash HD recall da memoria fotografica + attivazione neurale."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from organism.cognition.photographic_memory import PhotographicMemory


@dataclass
class MentalImage:
    key: str
    labels: list[str] = field(default_factory=list)
    hd_thumb: list[int] = field(default_factory=list)
    sketch_thumb: list[int] = field(default_factory=list)
    resolution: str = "hd"
    flash_ms: float = 0.0
    gist: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "labels": self.labels,
            "resolution": self.resolution,
            "flash_ms": round(self.flash_ms, 1),
            "gist": self.gist,
        }


class VisualImagination:
    """Riattiva pattern visivi — HD 30-50ms poi decay a sketch low-res."""

    HD_TTL_S = 0.045
    SKETCH_TTL_S = 8.0

    def __init__(self, photo_memory: PhotographicMemory | None = None) -> None:
        self.photo = photo_memory or PhotographicMemory()
        self._active: dict[str, tuple[MentalImage, float]] = {}
        self._last_flash: MentalImage | None = None

    def flash(
        self,
        *,
        features: dict[str, Any] | None = None,
        labels: list[str] | None = None,
        neuron_activation: float = 0.5,
    ) -> MentalImage | None:
        """Trigger → rendering HD breve → cache sketch."""
        t0 = time.perf_counter()
        feat = features or {}
        recalled = self.photo.recall(feat, limit=1)
        if recalled:
            fr = recalled[0]
            key = str(fr.get("key", ""))
            lbs = list(labels or fr.get("labels", []))[:8]
            thumb = list(fr.get("thumb", []))
            gist = dict(fr.get("gist", {}))
        else:
            key = str(feat.get("object_sig", "internal"))[:16]
            lbs = list(labels or [])[:8]
            thumb = []
            gist = {
                "color": feat.get("color", ""),
                "scene_type": feat.get("scene_type", ""),
            }

        if not key and not lbs:
            return None

        # HD phase — thumb completa se disponibile
        hd = thumb[:256] if thumb else []
        img = MentalImage(
            key=key or f"img_{int(time.time())}",
            labels=lbs,
            hd_thumb=hd,
            sketch_thumb=hd[:64] if hd else [],
            resolution="hd",
            flash_ms=(time.perf_counter() - t0) * 1000.0,
            gist=gist,
        )
        boost = max(0.2, min(1.0, neuron_activation))
        img.flash_ms = max(img.flash_ms, 30.0 / boost)

        now = time.time()
        self._active[img.key] = (img, now)
        self._last_flash = img
        return img

    def recall_sketch(self, key: str = "") -> MentalImage | None:
        """Sketch low-res dopo decay HD — sufficiente per reasoning."""
        now = time.time()
        self._decay(now)
        if key and key in self._active:
            img, t = self._active[key]
            return img
        if self._last_flash:
            return self._last_flash
        return None

    def active_labels(self) -> list[str]:
        now = time.time()
        self._decay(now)
        out: list[str] = []
        for img, _ in self._active.values():
            for lb in img.labels:
                if lb and lb not in out:
                    out.append(lb)
        return out[:10]

    def _decay(self, now: float) -> None:
        expired: list[str] = []
        for key, (img, t) in self._active.items():
            age = now - t
            if age > self.SKETCH_TTL_S:
                expired.append(key)
            elif age > self.HD_TTL_S and img.resolution == "hd":
                img.resolution = "sketch"
                img.sketch_thumb = img.hd_thumb[:64]
                img.hd_thumb = []
        for k in expired:
            del self._active[k]

    def stats(self) -> dict[str, Any]:
        return {"active_images": len(self._active), "has_flash": self._last_flash is not None}

    def to_dict(self) -> dict[str, Any]:
        return {"active_keys": list(self._active.keys())[-8:]}

    def load_dict(self, data: dict[str, Any]) -> None:
        pass  # stato effimero — non persistito
