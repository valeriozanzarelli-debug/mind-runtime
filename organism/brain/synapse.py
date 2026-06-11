"""Synaptic connection with plastic weight."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Synapse:
    pre_id: int
    post_id: int
    weight: float
    last_pre_spike: float = -1.0
    last_post_spike: float = -1.0

    def transmit(self, pre_activation: float) -> float:
        return pre_activation * self.weight
