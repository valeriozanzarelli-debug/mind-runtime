"""Connectivity map — inter-system wiring for the biological brain."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConnectionRule:
    source: str
    target: str
    connections_per_neuron: int
    weight_init: str = "xavier_uniform"
    bidirectional: bool = False
    pathway: str = ""
    plastic: bool = True
    dopamine_modulated: bool = False


# Feedforward: sensory → thalamus → integration → prefrontal → language → motor
# Feedback: prefrontal → sensory (prediction), dopamine → STDP modulation
CONNECTIVITY: tuple[ConnectionRule, ...] = (
    # ── Sensory → Thalamic relay (gating) ──
    ConnectionRule("auditory_cortex", "thalamic_relay", 12, pathway="sensory_gate"),
    ConnectionRule("visual_cortex", "thalamic_relay", 10, pathway="sensory_gate"),
    ConnectionRule("proprioceptive", "insula", 8, pathway="body_sense"),
    ConnectionRule("interoceptive", "insula", 15, pathway="homeostasis"),
    ConnectionRule("interoceptive", "amygdala", 6, pathway="urgency"),
    # ── Thalamic → Integration ──
    ConnectionRule("thalamic_relay", "association_cortex", 20, pathway="feedforward"),
    ConnectionRule("thalamic_relay", "temporal_lobe", 15, pathway="feedforward"),
    ConnectionRule("insula", "association_cortex", 12, pathway="emotion_meaning"),
    ConnectionRule("insula", "prefrontal_cortex", 8, pathway="intero_to_pfc"),
  # ── Integration ↔ Integration ──
    ConnectionRule("association_cortex", "temporal_lobe", 18, bidirectional=True, pathway="assoc_temporal"),
    ConnectionRule("temporal_lobe", "association_cortex", 18, pathway="assoc_temporal"),
    # ── Integration → Prefrontal + Limbic ──
    ConnectionRule("association_cortex", "prefrontal_cortex", 25, pathway="meaning_to_wm"),
    ConnectionRule("temporal_lobe", "prefrontal_cortex", 15, pathway="context_to_wm"),
    ConnectionRule("amygdala", "anterior_cingulate", 20, pathway="threat_conflict"),
    ConnectionRule("amygdala", "prefrontal_cortex", 10, pathway="emotion_override"),
    ConnectionRule("anterior_cingulate", "prefrontal_cortex", 15, bidirectional=True, pathway="conflict_loop"),
    ConnectionRule("prefrontal_cortex", "anterior_cingulate", 15, pathway="conflict_loop"),
    # ── Prediction error → Dopamine ──
    ConnectionRule("prefrontal_cortex", "dopamine_neurons", 12, pathway="prediction"),
    ConnectionRule("association_cortex", "dopamine_neurons", 8, pathway="outcome"),
    ConnectionRule("dopamine_neurons", "association_cortex", 10, dopamine_modulated=True, pathway="reward_learning"),
    ConnectionRule("dopamine_neurons", "prefrontal_cortex", 8, dopamine_modulated=True, pathway="reward_learning"),
    ConnectionRule("dopamine_neurons", "temporal_lobe", 6, dopamine_modulated=True, pathway="reward_learning"),
    # ── Prefrontal → Language ──
    ConnectionRule("prefrontal_cortex", "wernicke", 20, pathway="comprehension_context"),
    ConnectionRule("association_cortex", "wernicke", 15, pathway="semantic_input"),
    ConnectionRule("wernicke", "broca", 25, pathway="dorsal_stream"),
    ConnectionRule("broca", "motor_cortex", 20, pathway="articulation"),
    # ── Top-down feedback (prediction) ──
    ConnectionRule("prefrontal_cortex", "auditory_cortex", 8, pathway="top_down_pred"),
    ConnectionRule("prefrontal_cortex", "visual_cortex", 6, pathway="top_down_pred"),
    ConnectionRule("prefrontal_cortex", "thalamic_relay", 10, pathway="attention_bias"),
    # ── Consciousness loop ──
    ConnectionRule("thalamic_relay", "consciousness_integrator", 8, pathway="phi_input"),
    ConnectionRule("association_cortex", "consciousness_integrator", 6, pathway="phi_input"),
    ConnectionRule("prefrontal_cortex", "consciousness_integrator", 6, pathway="phi_input"),
    ConnectionRule("posterior_cingulate", "consciousness_integrator", 10, pathway="self_phi"),
    ConnectionRule("temporo_parietal_junction", "consciousness_integrator", 8, pathway="social_phi"),
    ConnectionRule("consciousness_integrator", "thalamic_relay", 5, pathway="phi_gate"),
    ConnectionRule("posterior_cingulate", "prefrontal_cortex", 12, bidirectional=True, pathway="self_awareness"),
    ConnectionRule("prefrontal_cortex", "posterior_cingulate", 12, pathway="self_awareness"),
    ConnectionRule("temporo_parietal_junction", "association_cortex", 10, bidirectional=True, pathway="theory_of_mind"),
    ConnectionRule("association_cortex", "temporo_parietal_junction", 10, pathway="theory_of_mind"),
)

INFORMATION_FLOW = [
    "sensory",
    "thalamic_relay",
    "integration",
    "prefrontal_limbic",
    "language",
    "motor_cortex",
]


def rules_for_region(region: str) -> list[ConnectionRule]:
    return [r for r in CONNECTIVITY if r.source == region or (r.bidirectional and r.target == region)]
