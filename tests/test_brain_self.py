"""Onde, sé, sogno — cervello disincarnato."""

from pathlib import Path

import pytest

from organism.autonomous.baby_agent import BabyAgent
from organism.brain.oscillation import inject_wave
from organism.cognition.dream import DreamEngine
from organism.cognition.self_model import SelfModel
from organism.cognition.waves import BrainWaveCycle
from organism.runtime import OrganismRuntime


@pytest.fixture
def baby_store(tmp_path: Path):
    return str(tmp_path / "baby.json")


def test_wave_cycle_rotates():
    w = BrainWaveCycle()
    phases = {w.advance().phase for _ in range(10)}
    assert "dream" in phases or "think" in phases


def test_inject_wave_modulates_brain():
    org = OrganismRuntime.baby(seed=1)
    before = org.brain.layer_activation_summary().get("sensory", 0)
    inject_wave(org.brain, "perceive", tick=3)
    after = org.brain.layer_activation_summary().get("sensory", 0)
    assert after >= before


def test_dream_on_idle():
    org = OrganismRuntime.baby(seed=2)
    from mind.types import Fragment

    org.memory.add(
        Fragment(id="m1", title="luce nel buio", weight=0.5, sensation_id="t", hooks=["luce"])
    )
    d = DreamEngine(seed=2)
    st = d.cycle(org.brain, org.memory, __import__("organism.teaching.words", fromlist=["WordLearner"]).WordLearner(), idle_s=20)
    assert st.active


def test_self_continuity_grows(baby_store):
    b = BabyAgent(seed=40, store_path=baby_store)
    b.birth()
    c0 = b.self_model.state.continuity
    for _ in range(5):
        b.brain_pulse_tick()
    assert b.self_model.state.continuity >= c0


def test_pulse_returns_wave_and_self(baby_store):
    b = BabyAgent(seed=41, store_path=baby_store)
    b.birth()
    r = b.brain_pulse_tick()
    assert r.get("wave", {}).get("phase")
    assert "continuity" in r.get("self", {})
