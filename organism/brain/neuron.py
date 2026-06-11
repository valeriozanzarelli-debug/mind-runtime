"""Single neuron unit — spike-capable node in the topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Neuron:
    id: int
    layer: str  # sensory | associative | motor
    subtype: str
    meta: dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0
    last_spike_t: float = -1.0
    spike_count: int = 0

    def receptive_field(self) -> tuple[int, int] | None:
        rf = self.meta.get("receptive_field")
        if rf is None:
            return None
        return int(rf[0]), int(rf[1])

    def frequency_band(self) -> tuple[float, float] | None:
        band = self.meta.get("frequency_band")
        if band is None:
            return None
        return float(band[0]), float(band[1])

    def fire(self, t: float, intensity: float = 1.0) -> None:
        self.activation = min(1.0, self.activation + intensity)
        self.last_spike_t = t
        self.spike_count += 1

    def decay(self, rate: float = 0.15) -> None:
        self.activation = max(0.0, self.activation - rate)

    def leak(self, rate: float = 0.15, *, floor: float = 0.01) -> bool:
        """Perdita efficiente — ritorna False se il neurone è spento."""
        if self.activation <= floor:
            self.activation = 0.0
            return False
        self.activation = max(0.0, self.activation - rate)
        if self.activation <= floor:
            self.activation = 0.0
            return False
        return True
