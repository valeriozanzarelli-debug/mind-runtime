"""Onde neurali — il grafo pulsa; il codice non è il pensiero."""

from __future__ import annotations

import math
from typing import Any

from organism.brain.topology import ACTIVE_FLOOR, NeuralTopology

PHASE_LAYERS: dict[str, dict[str, float]] = {
    "perceive": {"sensory": 0.35, "associative": 0.12, "motor": 0.05},
    "think": {"sensory": 0.1, "associative": 0.38, "motor": 0.08},
    "dream": {"sensory": 0.04, "associative": 0.28, "motor": 0.02},
    "reflect": {"sensory": 0.08, "associative": 0.42, "motor": 0.12},
    "rest": {"sensory": 0.06, "associative": 0.1, "motor": 0.03},
}

LARGE_BRAIN = 50_000
SPARSE_SAMPLE = 400


def inject_wave(
    brain: NeuralTopology,
    phase: str,
    *,
    tick: int,
    amplitude: float = 1.0,
) -> dict[str, float]:
    """Modula attivazione per fase d'onda — sparsa a scala mega."""
    profile = PHASE_LAYERS.get(phase, PHASE_LAYERS["rest"])
    wave = math.sin(tick * 0.17) * 0.5 + 0.5
    boosted: dict[str, float] = {}
    large = brain.neuron_count >= LARGE_BRAIN

    for layer, base in profile.items():
        gain = base * amplitude * (0.65 + 0.35 * wave)
        boosted[layer] = gain
        if large:
            _boost_layer_sparse(brain, layer, gain, phase, wave)
        else:
            _boost_layer_dense(brain, layer, gain, phase, wave)

    brain.propagate(steps=1)
    return boosted


def _boost_layer_dense(
    brain: NeuralTopology, layer: str, gain: float, phase: str, wave: float
) -> None:
    for n in brain.neurons.values():
        if n.layer != layer:
            continue
        n.activation = min(1.0, n.activation + gain * 0.08)
        if n.activation > ACTIVE_FLOOR:
            brain._active.add(n.id)
    if phase == "dream" and layer == "associative":
        for n in brain.get_neurons("associative", "memory_consolidator"):
            n.activation = min(1.0, n.activation + 0.12 * wave)
            if n.activation > ACTIVE_FLOOR:
                brain._active.add(n.id)
    elif phase == "reflect" and layer == "associative":
        for n in brain.get_neurons("associative", "pattern_matcher"):
            n.activation = min(1.0, n.activation + 0.1 * wave)
            if n.activation > ACTIVE_FLOOR:
                brain._active.add(n.id)


def _boost_layer_sparse(
    brain: NeuralTopology, layer: str, gain: float, phase: str, wave: float
) -> None:
    touched = 0
    for nid in list(brain._active):
        n = brain.neurons.get(nid)
        if n is None or n.layer != layer:
            continue
        n.activation = min(1.0, n.activation + gain * 0.08)
        touched += 1

    if touched >= 80:
        return

    # Campione sottile per layer — come oscillazione corticale diffusa
    for (ly, subtype), ids in brain._by_layer_subtype.items():
        if ly != layer or not ids:
            continue
        step = max(1, len(ids) // SPARSE_SAMPLE)
        for nid in ids[::step][:SPARSE_SAMPLE]:
            n = brain.neurons[nid]
            n.activation = min(1.0, n.activation + gain * 0.04)
            if n.activation > ACTIVE_FLOOR:
                brain._active.add(nid)

    if phase == "dream":
        ids = brain._by_layer_subtype.get(("associative", "memory_consolidator"), [])
        step = max(1, len(ids) // 200)
        for nid in ids[::step][:200]:
            n = brain.neurons[nid]
            n.activation = min(1.0, n.activation + 0.12 * wave)
            if n.activation > ACTIVE_FLOOR:
                brain._active.add(nid)
    elif phase == "reflect":
        ids = brain._by_layer_subtype.get(("associative", "pattern_matcher"), [])
        step = max(1, len(ids) // 200)
        for nid in ids[::step][:200]:
            n = brain.neurons[nid]
            n.activation = min(1.0, n.activation + 0.1 * wave)
            if n.activation > ACTIVE_FLOOR:
                brain._active.add(nid)
