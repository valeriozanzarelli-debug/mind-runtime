"""Single neuron unit — spike-capable node in the topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Neuron:
    id: int
    system: str
    region: str
    meta: dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0
    membrane_potential: float = -70.0
    last_spike_t: float = -1.0
    spike_count: int = 0
    prediction: float = 0.0

    def fire(self, t: float, intensity: float = 1.0) -> None:
        self.activation = min(1.0, self.activation + intensity)
        self.membrane_potential = min(40.0, self.membrane_potential + 30.0 * intensity)
        self.last_spike_t = t
        self.spike_count += 1

    def decay(self, rate: float = 0.12) -> None:
        self.activation = max(0.0, self.activation - rate)
        self.membrane_potential = max(-70.0, self.membrane_potential - 2.0)

    def leak(self, rate: float = 0.12, *, floor: float = 0.01) -> bool:
        if self.activation <= floor:
            self.activation = 0.0
            return False
        self.activation = max(0.0, self.activation - rate)
        if self.activation <= floor:
            self.activation = 0.0
            return False
        return True
