"""Ciclo ad onde — percepire, pensare, sognare, riflettere."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PHASES = ("perceive", "think", "dream", "reflect", "rest")
LABELS = {
    "perceive": "percepisce",
    "think": "pensa",
    "dream": "sogna",
    "reflect": "riflette",
    "rest": "riposo",
}


@dataclass
class WaveState:
    phase: str
    tick: int
    cycle: int
    amplitude: float

    @property
    def label(self) -> str:
        return LABELS.get(self.phase, self.phase)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "label": LABELS.get(self.phase, self.phase),
            "tick": self.tick,
            "cycle": self.cycle,
            "amplitude": round(self.amplitude, 3),
        }


class BrainWaveCycle:
    """Avanza fasi come ritmi corticali — non è un loop di output."""

    def __init__(self) -> None:
        self._idx = 0
        self._tick = 0
        self._cycles = 0
        self._last: WaveState | None = None

    def _phase_at(self, *, idle: bool) -> str:
        if idle:
            order = ("dream", "rest", "think", "perceive", "reflect")
            return order[self._idx % len(order)]
        return PHASES[self._idx % len(PHASES)]

    def _amplitude(self, phase: str, arousal: float) -> float:
        amp = 0.55 + 0.35 * arousal
        if phase == "dream":
            amp *= 1.1
        return amp

    @property
    def last(self) -> WaveState:
        if self._last is None:
            return self.advance(idle=False)
        return self._last

    def advance(self, *, idle: bool = False, arousal: float = 0.5) -> WaveState:
        self._tick += 1
        phase = self._phase_at(idle=idle)
        self._idx += 1
        if self._idx % len(PHASES) == 0:
            self._cycles += 1
        self._last = WaveState(
            phase=phase,
            tick=self._tick,
            cycle=self._cycles,
            amplitude=self._amplitude(phase, arousal),
        )
        return self._last

    def stats(self) -> dict[str, Any]:
        return {"tick": self._tick, "cycles": self._cycles, "idx": self._idx}
