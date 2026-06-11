"""Percorsi appresi — attivazione sinaptica, non lookup di stringhe."""

from __future__ import annotations

import re
from typing import Any

from organism.brain.growth import active_ids, wire_coactive
from organism.brain.topology import NeuralTopology
from organism.motor.emergent_speech import EmergentSpeechMotor


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zàèéìòù']+", text.lower()) if len(w) > 1]


def wire_dialogue_pathway(
    brain: NeuralTopology,
    motor: EmergentSpeechMotor,
    *,
    when: str,
    say: str,
) -> int:
    """Hebb: stimolo udito → memoria associativa → motore fonetico della risposta."""
    when_t = _tokens(when)
    say_t = _tokens(say)
    if not when_t or not say_t:
        return 0
    sensory = active_ids(brain, "sensory", "text_semantic_encoder", min_act=0.02)
    if not sensory:
        sensory = [n.id for n in brain.get_neurons("sensory", "text_semantic_encoder")[:12]]
    mem = active_ids(brain, "associative", "memory_consolidator", min_act=0.02)
    if not mem:
        mem = [n.id for n in brain.get_neurons("associative", "memory_consolidator")[:10]]
    motor_ids = [n.id for n in motor.phoneme_neurons[: max(8, len(say_t) * 2)]]
    grown = wire_coactive(brain, sensory[:14], mem[:12], max_new=8, weight=0.24)
    grown += wire_coactive(brain, mem[:12], motor_ids, max_new=10, weight=0.26)
    return grown


def prime_dialogue_pathway(
    brain: NeuralTopology,
    motor: EmergentSpeechMotor,
    *,
    when: str,
    say: str,
    boost: float = 0.45,
) -> None:
    """Riattiva il percorso insegnato — la risposta emerge dal motore, non da template."""
    motor.hear(when, boost=boost * 0.85)
    motor.hear(say, boost=boost * 1.1)
    for n in brain.get_neurons("sensory", "text_semantic_encoder")[:14]:
        n.activation = min(1.0, n.activation + boost * 0.55)
    for n in brain.get_neurons("associative", "pattern_matcher")[:10]:
        n.activation = min(1.0, n.activation + boost * 0.4)
    for n in brain.get_neurons("associative", "memory_consolidator")[:10]:
        n.activation = min(1.0, n.activation + boost * 0.5)
    for n in motor.phoneme_neurons[:16]:
        n.activation = min(1.0, n.activation + boost * 0.48)
    brain.propagate(steps=3)


def prime_curiosity_pathway(
    brain: NeuralTopology,
    motor: EmergentSpeechMotor,
    lexicon: Any,
    *,
    focus: str,
    heard: str = "",
    boost: float = 0.5,
) -> None:
    """Curiosità — riattiva percorsi lessicali già appresi, non inietta frasi fisse."""
    if heard.strip():
        motor.hear(heard, boost=boost * 0.85)
    focus_l = focus.lower().strip()
    if focus_l:
        motor.hear(focus_l, boost=boost * 1.15)
        lexicon.prime_word(focus_l, boost=boost * 1.1)
    ranked = lexicon.ranked([focus_l] + lexicon.known_words(20)) if focus_l else lexicon.ranked(lexicon.known_words(20))
    for w in ranked[:10]:
        lexicon.prime_word(w, boost=boost * 0.65)
    for n in brain.get_neurons("associative", "pattern_matcher")[:12]:
        n.activation = min(1.0, n.activation + boost * 0.42)
    for n in brain.get_neurons("associative", "memory_consolidator")[:10]:
        n.activation = min(1.0, n.activation + boost * 0.48)
    for n in motor.phoneme_neurons[:18]:
        n.activation = min(1.0, n.activation + boost * 0.38)
    brain.propagate(steps=3)
