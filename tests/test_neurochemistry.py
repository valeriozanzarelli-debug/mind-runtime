"""Test neurochimica — modulazione dopamina/stress e plasticità."""

from organism.cognition.neurochemistry import NeurochemistryEngine


def test_dopamine_rises_on_reward():
    nc = NeurochemistryEngine()
    before = nc.state.dopamine
    nc.tick(joy=0.9, trust=0.8, curiosity=0.7, learned=True)
    assert nc.state.dopamine > before


def test_stress_raises_norepinephrine():
    nc = NeurochemistryEngine()
    nc.tick(fear=0.85, anger=0.5, stress=0.7)
    assert nc.state.norepinephrine > 0.4


def test_plasticity_gain_bounded():
    nc = NeurochemistryEngine()
    nc.tick(curiosity=0.9, learned=True)
    gain = nc.plasticity_gain()
    assert 0.5 <= gain <= 2.0


def test_modulate_affect_dims():
    nc = NeurochemistryEngine()
    nc.tick(joy=0.8, trust=0.7)
    j, f, _, _, t, c = nc.modulate_affect_dims(
        joy=0.3, fear=0.2, sadness=0.1, anger=0.0, trust=0.4, curiosity=0.5
    )
    assert j >= 0.3
    assert t >= 0.4
    assert c >= 0.5
