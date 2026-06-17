"""Lettura stato dal grafo neurale — niente etichette hardcoded sul comportamento."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from organism.timezone_util import TZ


@dataclass
class BrainMood:
    layers: dict[str, float]
    synapses: int
    synapses_grown: int
    motor_pressure: float
    inhibition: float
    arousal: float
    wants_voice: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": {k: round(v, 3) for k, v in self.layers.items()},
            "synapses": self.synapses,
            "synapses_grown": self.synapses_grown,
            "motor_pressure": round(self.motor_pressure, 3),
            "inhibition": round(self.inhibition, 3),
            "arousal": round(self.arousal, 3),
            "wants_voice": self.wants_voice,
        }


def inject_circadian(brain, hour: int | None = None) -> float:
    """Bias di arousal nel grafo — orario come input debole, non regola fissa."""
    if hour is None:
        hour = datetime.now(TZ).hour
    # curva liscia: picco pomeriggio, minimo notte
    arousal = max(0.05, min(1.0, 0.35 + 0.65 * _circadian_wave(hour)))
    assoc = brain.get_neurons("associative", "pattern_matcher")
    n = max(1, int(len(assoc) * arousal * 0.08))
    for neuron in assoc[:n]:
        neuron.activation = min(1.0, neuron.activation + arousal * 0.06)
    return arousal


def read_brain_mood(
    brain,
    *,
    synapses_at_birth: int,
    motor_pressure: float,
    inhibition: float,
    wants_voice: bool,
    arousal: float,
) -> BrainMood:
    layers = brain.layer_activation_summary()
    return BrainMood(
        layers=layers,
        synapses=brain.synapse_count,
        synapses_grown=max(0, brain.synapse_count - synapses_at_birth),
        motor_pressure=motor_pressure,
        inhibition=inhibition,
        arousal=arousal,
        wants_voice=wants_voice,
    )


def _circadian_wave(hour: int) -> float:
    import math

    # 14:00 ≈ 1.0, 3:00 ≈ 0.0
    return (math.sin((hour - 3) / 24 * 2 * math.pi) + 1) / 2
