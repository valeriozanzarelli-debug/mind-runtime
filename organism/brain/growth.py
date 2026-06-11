"""Crescita sinaptica — nuove connessioni da co-attivazione (Hebb emergente)."""

from __future__ import annotations

import os

from organism.brain.topology import NeuralTopology

_VARIANT = os.environ.get("ORGANISM_DNA_VARIANT", "baby")
_SUPERHUMAN = _VARIANT in ("superhuman", "genesis")


def _scale_max(base: int) -> int:
    """Superhuman: sinapsi illimitate — burst più ampi ad ogni co-attivazione."""
    return base * 3 if _SUPERHUMAN else base


def wire_coactive(
    brain: NeuralTopology,
    pre_ids: list[int],
    post_ids: list[int],
    *,
    max_new: int = 10,
    weight: float = 0.18,
) -> int:
    """Nuove sinapsi tra neuroni co-attivi; rinforza quelle esistenti."""
    if not pre_ids or not post_ids:
        return 0
    created = 0
    for pre in pre_ids:
        for post in post_ids:
            if brain.has_edge(pre, post):
                for syn in brain.outgoing.get(pre, []):
                    if syn.post_id == post:
                        syn.weight = min(1.0, syn.weight + weight * 0.5)
                continue
            if created >= max_new:
                continue
            brain.connect(pre, post, weight=weight, weight_init="small_random")
            created += 1
    return created


def active_ids(brain: NeuralTopology, layer: str, subtype: str, *, min_act: float = 0.22) -> list[int]:
    return [n.id for n in brain.get_neurons(layer, subtype) if n.activation >= min_act]


def wire_motor_to_sensory(
    brain: NeuralTopology,
    *,
    motor_subtype: str = "speech_phoneme_generator",
    max_new: int = 5,
) -> int:
    """Stream dorsale — collega motore fonetico a codificatori sensoriali (loop DIVA)."""
    pre = active_ids(brain, "motor", motor_subtype, min_act=0.12)
    post_text = active_ids(brain, "sensory", "text_semantic_encoder", min_act=0.08)
    post_audio = active_ids(brain, "sensory", "audio_frequency_analyzer", min_act=0.08)
    post = (post_text + post_audio)[:14]
    if not pre or not post:
        return 0
    return wire_coactive(brain, pre[:10], post, max_new=max_new, weight=0.16)


def wire_self_recurrence(brain: NeuralTopology, *, max_new: int = 4) -> int:
    """Loop sé — memoria ↔ emozione (senso di continuità interna)."""
    mem = active_ids(brain, "associative", "memory_consolidator", min_act=0.05)
    emo = active_ids(brain, "associative", "emotion_modulator", min_act=0.05)
    pat = active_ids(brain, "associative", "pattern_matcher", min_act=0.05)
    grown = wire_coactive(brain, mem[:8], emo[:8], max_new=max_new, weight=0.14)
    grown += wire_coactive(brain, pat[:6], mem[:6], max_new=3, weight=0.12)
    grown += wire_coactive(brain, emo[:6], pat[:6], max_new=3, weight=0.11)
    return grown


def wire_vision_only(brain: NeuralTopology, *, max_new: int = 4) -> int:
    """Vista senza testo — edge → pattern (percezione continua)."""
    vision = active_ids(brain, "sensory", "vision_edge_detector", min_act=0.08)
    assoc = active_ids(brain, "associative", "pattern_matcher", min_act=0.05)
    if not vision:
        vision = [n.id for n in brain.get_neurons("sensory", "vision_edge_detector")[:10]]
    if not assoc:
        assoc = [n.id for n in brain.get_neurons("associative", "pattern_matcher")[:10]]
    return wire_coactive(brain, vision[:10], assoc[:10], max_new=max_new, weight=0.16)


def wire_cross_modal(
    brain: NeuralTopology,
    *,
    had_vision: bool,
    had_text: bool,
    max_new: int = 10,
) -> int:
    """Binding intermodale vista↔linguaggio (Bahrick / IFC)."""
    grown = 0
    if had_vision and not had_text:
        return wire_vision_only(brain, max_new=max_new)
    if not (had_vision and had_text):
        return 0
    vision = active_ids(brain, "sensory", "vision_edge_detector", min_act=0.1)
    text = active_ids(brain, "sensory", "text_semantic_encoder", min_act=0.1)
    assoc = active_ids(brain, "associative", "pattern_matcher", min_act=0.08)
    motor = [n.id for n in brain.get_neurons("motor", "speech_phoneme_generator")][:10]
    grown = 0
    grown += wire_coactive(brain, vision[:10], text[:10], max_new=max_new, weight=0.22)
    grown += wire_coactive(brain, vision[:8], assoc[:10], max_new=max(4, max_new // 2), weight=0.19)
    grown += wire_coactive(brain, assoc[:10], motor, max_new=max(4, max_new // 2), weight=0.17)
    grown += wire_coactive(brain, text[:8], motor, max_new=4, weight=0.16)
    return grown


def wire_teach_burst(brain: NeuralTopology, *, had_vision: bool, had_text: bool) -> int:
    """Burst sinaptico all'insegnamento — più connessioni quando impara."""
    grown = wire_cross_modal(brain, had_vision=had_vision, had_text=had_text, max_new=_scale_max(14))
    assoc = active_ids(brain, "associative", "pattern_matcher", min_act=0.05)
    mem = active_ids(brain, "associative", "memory_consolidator", min_act=0.05)
    motor = [n.id for n in brain.get_neurons("motor", "speech_phoneme_generator")]
    grown += wire_coactive(brain, assoc[:18], mem[:14], max_new=_scale_max(8), weight=0.2)
    grown += wire_coactive(brain, mem[:14], motor[:16], max_new=_scale_max(8), weight=0.18)
    if _SUPERHUMAN:
        vision = active_ids(brain, "sensory", "vision_edge_detector", min_act=0.04)
        grown += wire_coactive(brain, vision[:16], mem[:12], max_new=20, weight=0.17)
    return grown


def depress_synapses(brain: NeuralTopology, neuron_ids: list[int], *, amount: float = 0.03) -> int:
    """Depressione sinaptica su errori motori — impara dagli errori."""
    targets = set(neuron_ids)
    adjusted = 0
    seen: set[int] = set()
    for nid in targets:
        if nid in seen:
            continue
        seen.add(nid)
        for syn in brain.outgoing.get(nid, ()):
            if syn.post_id in targets or syn.pre_id in targets:
                syn.weight = max(0.02, syn.weight - amount)
                adjusted += 1
        for syn in brain.incoming.get(nid, ()):
            if syn.pre_id in targets:
                syn.weight = max(0.02, syn.weight - amount)
                adjusted += 1
    return adjusted
