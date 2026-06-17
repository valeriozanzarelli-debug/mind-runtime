"""Integrazione cervello umano completo — endocrino + psyche/superego nel Baby."""

from organism.autonomous.baby_agent import BabyAgent
from organism.cognition.endocrine import EndocrineSystem
from organism.cognition.interoception import InteroceptionEngine


def test_endocrine_hpa_stress():
    endo = EndocrineSystem()
    stress = endo.stress_from_affect(fear=0.8, anger=0.3, shame=0.4)
    endo.tick(stress=stress, hour=14)
    assert endo.hormones.cortisol > 0.25
    assert endo.hormones.adrenaline > 0.15


def test_interoception_links_chemistry():
    from organism.cognition.neurochemistry import NeurochemistryEngine

    nc = NeurochemistryEngine()
    nc.tick(fear=0.7, stress=0.6)
    endo = EndocrineSystem()
    endo.tick(stress=0.6, hour=3)
    intero = InteroceptionEngine()
    intero.update(nc.state, endo.hormones, fear=0.7, shame=0.2, idle_s=90)
    assert intero.state.heart_rate > 0.4
    assert intero.state.label in ("ansia_viscerale", "neutro", "stanco", "a_mio_agio")


def test_baby_agent_has_human_subsystems():
    agent = BabyAgent(seed=7)
    assert agent.psyche is not None
    assert agent.superego is not None
    assert agent.neurochemistry is not None
    assert agent.endocrine is not None
    assert agent.body_schema is not None
    assert agent.quantum_layer is not None


def test_baby_birth_wires_motion():
    agent = BabyAgent(seed=11)
    snap = agent.birth()
    assert snap.get("born") is True
    assert agent.motion is not None
    pulse = agent.brain_pulse_tick()
    assert pulse.get("alive") is True
    assert "neurochemistry" in pulse
    assert "body_schema" in pulse
