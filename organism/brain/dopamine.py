"""Dopamine system — prediction error drives STDP modulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DopamineSystem:
    """Mesolimbic prediction-error signal from prefrontal vs sensory mismatch."""

    baseline: float = 0.3
    level: float = 0.3
    prediction_error: float = 0.0
    history: list[float] = field(default_factory=list)
    decay_rate: float = 0.05

    def compute_prediction_error(
        self,
        prefrontal_activation: float,
        sensory_activation: float,
        *,
        prefrontal_prediction: float = 0.0,
        sensory_input: float = 0.0,
    ) -> float:
        """δ = actual - predicted (reward prediction error analog)."""
        predicted = prefrontal_prediction if prefrontal_prediction > 0 else prefrontal_activation * 0.8
        actual = sensory_input if sensory_input > 0 else sensory_activation
        self.prediction_error = actual - predicted
        return self.prediction_error

    def update(self, prediction_error: float, dopamine_neuron_activity: float = 0.0) -> float:
        """Phasic dopamine burst on positive PE, dip on negative."""
        phasic = np.tanh(prediction_error * 2.0) * 0.4
        tonic = dopamine_neuron_activity * 0.3
        self.level = max(0.0, min(1.0, self.baseline + phasic + tonic))
        self.level = self.level * (1.0 - self.decay_rate) + self.baseline * self.decay_rate
        self.history.append(self.level)
        if len(self.history) > 200:
            self.history = self.history[-200:]
        return self.level

    def stdp_modulation(self) -> float:
        """Returns multiplier for STDP learning rate [0.5, 1.5]."""
        return 0.5 + self.level

    def to_dict(self) -> dict:
        return {
            "level": round(self.level, 4),
            "prediction_error": round(self.prediction_error, 4),
            "baseline": self.baseline,
        }
