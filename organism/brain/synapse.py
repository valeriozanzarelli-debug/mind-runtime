"""Synaptic connection between neurons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Synapse:
    pre_id: int
    post_id: int
    weight: float
    delay_ms: float = 1.0
    plastic: bool = True
    pathway: str = ""
    dopamine_modulated: bool = False

    def transmit(self, pre_activation: float) -> float:
        return pre_activation * self.weight
