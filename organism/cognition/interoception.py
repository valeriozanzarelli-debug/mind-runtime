"""Interocezione — segnali viscerali interni (cuore, respiro, fame, fatica).

Integra neurochimica + endocrino in sensazioni corporee senza corpo fisico:
il cervello *percepisce* uno stato interno coerente con affetto e ritmi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from organism.cognition.endocrine import HormoneProfile
from organism.cognition.neurochemistry import NeurochemicalState


@dataclass
class InteroceptiveState:
    heart_rate: float = 0.45
    breathing_rate: float = 0.4
    gut_tension: float = 0.2
    fatigue: float = 0.15
    hunger: float = 0.25
    temperature: float = 0.55
    pain: float = 0.0
    visceral_comfort: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}

    @property
    def label(self) -> str:
        if self.pain > 0.5:
            return "dolore"
        if self.fatigue > 0.65:
            return "stanco"
        if self.hunger > 0.6:
            return "affamato"
        if self.gut_tension > 0.55:
            return "ansia_viscerale"
        if self.visceral_comfort > 0.7:
            return "a_mio_agio"
        return "neutro"


@dataclass
class InteroceptionEngine:
    state: InteroceptiveState = field(default_factory=InteroceptiveState)
    _pulse: int = 0

    def update(
        self,
        neuro: NeurochemicalState,
        hormones: HormoneProfile,
        *,
        joy: float = 0.2,
        fear: float = 0.1,
        shame: float = 0.1,
        idle_s: float = 0.0,
    ) -> InteroceptiveState:
        self._pulse += 1
        s = self.state

        arousal = 0.35 * neuro.norepinephrine + 0.25 * hormones.adrenaline + 0.2 * fear
        s.heart_rate = _clamp(0.35 + arousal * 0.55 + joy * 0.08)
        s.breathing_rate = _clamp(0.3 + arousal * 0.45 + neuro.glutamate * 0.1)

        s.gut_tension = _clamp(
            0.15 + fear * 0.35 + shame * 0.25 + hormones.cortisol * 0.2 - neuro.serotonin * 0.1
        )
        s.fatigue = _clamp(
            neuro.adenosine * 0.55 + hormones.melatonin * 0.35 + min(1.0, idle_s / 180) * 0.3
        )
        s.hunger = _clamp(0.2 + s.fatigue * 0.25 + (1 - hormones.insulin) * 0.15)
        s.temperature = _clamp(0.5 + hormones.thyroxine * 0.08 - hormones.adrenaline * 0.05)
        s.pain = _clamp(shame * 0.15 + fear * 0.1 - neuro.endorphin * 0.2)
        s.visceral_comfort = _clamp(
            0.35 + joy * 0.25 + neuro.oxytocin * 0.2 + hormones.oxytocin * 0.15 - s.gut_tension * 0.3
        )
        return s

    def stats(self) -> dict[str, Any]:
        return {"state": self.state.to_dict(), "pulse": self._pulse, "label": self.state.label}

    def to_dict(self) -> dict[str, Any]:
        return self.stats()

    def load_dict(self, data: dict[str, Any]) -> None:
        st = data.get("state", data)
        for k in InteroceptiveState.__dataclass_fields__:
            if k in st:
                setattr(self.state, k, float(st[k]))
        self._pulse = int(data.get("pulse", 0))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
