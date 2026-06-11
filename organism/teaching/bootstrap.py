"""Bootstrap genesis — sa parlare e pensare, zero informazioni sul mondo."""

from __future__ import annotations

import itertools
from typing import Any

from organism.brain.growth import wire_cross_modal, wire_self_recurrence, wire_teach_burst
from organism.brain.topology import NeuralTopology
from organism.motor.emergent_speech import EmergentSpeechMotor


# Inventario fonologico italiano — capacità articolatoria, non lessico semantico
_CONSONANTS = "bcdfglmnprstvz"
_VOWELS = "aeiou"


def phonological_inventory() -> list[str]:
    """Sillabe possibili — come avere cordi vocali, non vocabolario."""
    syls: list[str] = []
    for c, v in itertools.product(_CONSONANTS, _VOWELS):
        syls.append(f"{c}{v}")
    for v in _VOWELS:
        syls.append(v)
    return syls[:80]


def wire_genesis_capacity(
    brain: NeuralTopology,
    speech: EmergentSpeechMotor,
    *,
    composer: Any | None = None,
) -> dict[str, Any]:
    """Cablaggio innato — loop sensorimotori senza insegnare fatti."""
    grown = 0
    grown += wire_teach_burst(brain, had_vision=True, had_text=True)
    grown += wire_self_recurrence(brain)
    grown += wire_cross_modal(brain, had_vision=True, had_text=True, max_new=18)

    motor = speech.phoneme_neurons[:24]
    for n in motor:
        n.activation = min(1.0, n.activation + 0.12)

    # Fonologia — esposizione minima (articolazione, non significato)
    inv = phonological_inventory()
    for chunk in (inv[:20], inv[20:40], inv[40:60]):
        blob = " ".join(chunk)
        speech.hear(blob, boost=0.08)

    brain.propagate(steps=3)
    return {
        "wires_grown": grown,
        "phonology_slots": len(inv),
        "motor_primed": len(motor),
    }
