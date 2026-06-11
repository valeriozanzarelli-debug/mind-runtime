"""Percorsi visivi — associa forma+colore a lessico e motore (Hebb)."""

from __future__ import annotations

from typing import Any

from organism.brain.growth import active_ids, wire_coactive
from organism.brain.topology import NeuralTopology
from organism.motor.emergent_speech import EmergentSpeechMotor


def wire_visual_association(
    brain: NeuralTopology,
    motor: EmergentSpeechMotor,
    *,
    name: str,
    color: str = "",
    boost: float = 0.5,
) -> int:
    """Insegna: pattern visivo attivo → memoria → fonemi del nome/colore."""
    vision = active_ids(brain, "sensory", "vision_edge_detector", min_act=0.08)
    if not vision:
        vision = [n.id for n in brain.get_neurons("sensory", "vision_edge_detector")[:16]]
    mem = active_ids(brain, "associative", "memory_consolidator", min_act=0.05)
    if not mem:
        mem = [n.id for n in brain.get_neurons("associative", "memory_consolidator")[:12]]
    text_enc = [n.id for n in brain.get_neurons("sensory", "text_semantic_encoder")[:14]]
    motor_ids = [n.id for n in motor.phoneme_neurons[:16]]

    phrase = name
    if color and color not in ("grigio", "colore"):
        phrase = f"{name} {color}"
    motor.hear(phrase, boost=boost * 1.1)

    grown = wire_coactive(brain, vision[:14], mem[:12], max_new=10, weight=0.22 * boost)
    grown += wire_coactive(brain, mem[:12], text_enc[:12], max_new=8, weight=0.2 * boost)
    grown += wire_coactive(brain, text_enc[:12], motor_ids, max_new=10, weight=0.24 * boost)
    for n in brain.get_neurons("associative", "pattern_matcher")[:10]:
        n.activation = min(1.0, n.activation + boost * 0.35)
    brain.propagate(steps=2)
    return grown


def prime_visual_percept(
    brain: NeuralTopology,
    motor: EmergentSpeechMotor,
    lexicon: Any,
    *,
    themes: list[str],
    boost: float = 0.45,
) -> None:
    """Riattiva lessico visivo appreso prima di parlare."""
    for w in themes[:8]:
        lexicon.prime_word(w, boost=boost * 0.7)
        motor.hear(w, boost=boost * 0.5)
    for n in brain.get_neurons("sensory", "vision_edge_detector")[:14]:
        n.activation = min(1.0, n.activation + boost * 0.4)
    for n in brain.get_neurons("associative", "pattern_matcher")[:10]:
        n.activation = min(1.0, n.activation + boost * 0.35)
    brain.propagate(steps=2)
