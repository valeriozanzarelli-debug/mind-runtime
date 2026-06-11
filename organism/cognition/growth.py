"""Crescita graduale del cervello — solo pensiero, senza corpo."""

from __future__ import annotations

import os
from typing import Any

from organism.brain.topology import NeuralTopology

_variant = os.environ.get("ORGANISM_DNA_VARIANT", "baby")
_default_max = {
    "genesis": 40000,
    "superhuman": 40000,
    "adolescent": 35000,
    "two_k": 25000,
    "compact": 50000,
    "mega": 45000,
    "giga": 50000,
}.get(_variant, 20000)
MAX_NEURONS = int(os.environ.get("ORGANISM_MAX_NEURONS", str(_default_max)))
GROWTH_BATCH = int(
    os.environ.get(
        "ORGANISM_GROWTH_BATCH",
        "200"
        if _variant in ("superhuman", "genesis")
        else ("120" if _variant in ("adolescent", "compact") else "100"),
    )
)
GROWTH_EVERY_PULSES = int(
    os.environ.get(
        "ORGANISM_GROWTH_EVERY",
        "12"
        if _variant in ("superhuman", "genesis")
        else ("20" if _variant == "adolescent" else "25"),
    )
)


class BrainGrowth:
    def __init__(self) -> None:
        self.pulses = 0
        self.growth_events = 0
        self.last_neurons = 0

    def maybe_grow(self, brain: NeuralTopology, *, pulses: int) -> dict[str, Any]:
        self.pulses = pulses
        n = brain.neuron_count
        if n >= MAX_NEURONS:
            return {"grown": 0, "reason": "cap"}
        if pulses <= 0 or pulses % GROWTH_EVERY_PULSES != 0:
            return {"grown": 0}
        before = n
        brain.add_neurons_bulk("associative", "pattern_matcher", GROWTH_BATCH)
        brain.add_neurons_bulk("associative", "memory_consolidator", max(12, GROWTH_BATCH // 5))
        brain.add_neurons_bulk("associative", "emotion_modulator", max(8, GROWTH_BATCH // 8))
        from organism.brain.growth import wire_coactive

        wire_cap = 24 if _variant == "superhuman" else 6
        pat = [i for i in brain._by_layer_subtype.get(("associative", "pattern_matcher"), [])][-GROWTH_BATCH:]
        mem = [i for i in brain._by_layer_subtype.get(("associative", "memory_consolidator"), [])][-16:]
        emo = [i for i in brain._by_layer_subtype.get(("associative", "emotion_modulator"), [])][-10:]
        wire_coactive(brain, pat[:28], mem[:16], max_new=wire_cap, weight=0.15)
        wire_coactive(brain, mem[:12], emo[:10], max_new=max(4, wire_cap // 2), weight=0.13)
        self.growth_events += 1
        self.last_neurons = brain.neuron_count
        return {
            "grown": brain.neuron_count - before,
            "neurons": brain.neuron_count,
            "events": self.growth_events,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "pulses": self.pulses,
            "growth_events": self.growth_events,
            "max_neurons": MAX_NEURONS,
            "last_neurons": self.last_neurons,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self.growth_events = int(data.get("growth_events", 0))
        self.last_neurons = int(data.get("last_neurons", 0))
