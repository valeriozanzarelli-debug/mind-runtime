"""Sistema endocrino — ghiandole e ormoni che modulano ritmi, stress, crescita, legame.

Assi modellati:
  - HPA (ipotalamo → ipofisi → surrenale → cortisolo/adrenalina)
  - Circadiano (epifisi → melatonina)
  - Metabolico / crescita (tiroide, GH, insulina — astratto)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GlandActivity:
    hypothalamus: float = 0.5
    pituitary: float = 0.45
    adrenal: float = 0.3
    pineal: float = 0.4
    thyroid: float = 0.5
    pancreas: float = 0.45
    gonads: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


@dataclass
class HormoneProfile:
    cortisol: float = 0.25
    melatonin: float = 0.2
    adrenaline: float = 0.15
    growth_hormone: float = 0.4
    thyroxine: float = 0.5
    insulin: float = 0.45
    oxytocin: float = 0.3
    testosterone: float = 0.35
    estrogen: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


@dataclass
class EndocrineSystem:
    glands: GlandActivity = field(default_factory=GlandActivity)
    hormones: HormoneProfile = field(default_factory=HormoneProfile)
    _pulse: int = 0

    def tick(
        self,
        *,
        hour: int = 12,
        wave_phase: str = "think",
        stress: float = 0.0,
        social_bond: float = 0.3,
        fatigue: float = 0.0,
        age_factor: float = 0.15,
    ) -> HormoneProfile:
        self._pulse += 1
        g = self.glands
        h = self.hormones

        # Circadiano → epifisi
        night = hour < 6 or hour >= 22
        circ = _circadian(hour)
        g.pineal = _clamp(0.3 + (0.65 if night else 0.1) * (1 - circ))
        h.melatonin = _clamp(h.melatonin * 0.92 + g.pineal * (0.12 if night else 0.03))

        # HPA — minaccia / correzione / paura
        g.hypothalamus = _clamp(0.35 + stress * 0.55)
        g.pituitary = _clamp(g.pituitary * 0.9 + g.hypothalamus * 0.12)
        g.adrenal = _clamp(g.adrenal * 0.88 + g.pituitary * stress * 0.2)
        h.cortisol = _clamp(h.cortisol * 0.94 + g.adrenal * stress * 0.08)
        h.adrenaline = _clamp(h.adrenaline * 0.9 + stress * 0.12)

        # Sonno profondo / REM → GH
        if wave_phase in ("dream", "rest"):
            h.growth_hormone = _clamp(h.growth_hormone + 0.06)
        h.growth_hormone = _clamp(h.growth_hormone * 0.98 + age_factor * 0.01)

        # Tiroide → metabolismo cognitivo
        g.thyroid = _clamp(0.4 + circ * 0.35)
        h.thyroxine = _clamp(h.thyroxine * 0.95 + g.thyroid * 0.06)

        # Pancreas / energia (fame astratta)
        g.pancreas = _clamp(0.35 + fatigue * 0.3)
        h.insulin = _clamp(h.insulin * 0.96 + (1 - fatigue) * 0.04)

        # Legame sociale → ipotalamo
        h.oxytocin = _clamp(h.oxytocin * 0.94 + social_bond * 0.08)

        # Gonadi — livello basale modulato da età simulata
        g.gonads = _clamp(0.25 + age_factor * 0.5)
        h.testosterone = _clamp(h.testosterone * 0.99 + g.gonads * 0.005)
        h.estrogen = _clamp(h.estrogen * 0.99 + g.gonads * 0.004)

        return h

    def stress_from_affect(self, *, fear: float, anger: float, shame: float) -> float:
        return _clamp(0.5 * fear + 0.35 * anger + 0.25 * shame)

    def circadian_arousal(self, hour: int) -> float:
        return _circadian(hour)

    def sleep_pressure(self) -> float:
        return _clamp(self.hormones.melatonin * 0.6 + self.hormones.cortisol * -0.15 + 0.2)

    def stats(self) -> dict[str, Any]:
        return {
            "glands": self.glands.to_dict(),
            "hormones": self.hormones.to_dict(),
            "pulse": self._pulse,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.stats()

    def load_dict(self, data: dict[str, Any]) -> None:
        g = data.get("glands", {})
        h = data.get("hormones", {})
        for k in GlandActivity.__dataclass_fields__:
            if k in g:
                setattr(self.glands, k, float(g[k]))
        for k in HormoneProfile.__dataclass_fields__:
            if k in h:
                setattr(self.hormones, k, float(h[k]))
        self._pulse = int(data.get("pulse", 0))


def _circadian(hour: int) -> float:
    return (math.sin((hour - 3) / 24 * 2 * math.pi) + 1) / 2


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
