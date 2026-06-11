"""Loop sensorimotorio — auto-ascolto e apprendimento dagli errori."""

from pathlib import Path

import pytest

from organism.autonomous.baby_agent import BabyAgent
from organism.cognition.speech_loop import SpeechSensorimotorLoop, syllable_similarity
from organism.runtime import OrganismRuntime


@pytest.fixture
def baby_store(tmp_path: Path):
    return str(tmp_path / "baby.json")


def test_syllable_similarity_perfect():
    assert syllable_similarity(["cia", "o"], ["cia", "o"]) == 1.0


def test_syllable_similarity_partial():
    sim = syllable_similarity(["ma", "ma"], ["ma", "pa"])
    assert 0.0 < sim < 1.0


def test_flow_without_forced_loop(baby_store):
    b = BabyAgent(seed=11, store_path=baby_store)
    b.birth()
    for _ in range(3):
        b.teach_repetition("mamma")
    tick = b.flow()
    assert "consciousness" in tick["moment"]


def test_error_penalizes_wrong_syllables(baby_store):
    b = BabyAgent(seed=12, store_path=baby_store)
    b.birth()
    b.speech.phonemes.hear("mamma papa", boost=2.0)
    w_before = dict(b.speech.phonemes._weights)
    loop = SpeechSensorimotorLoop()
    org = OrganismRuntime.baby(seed=12)
    from organism.cognition.motor_plan import MotorPlan

    plan = MotorPlan(text="ma-pa", syllables=["ma", "pa"], from_motor=True)
    loop.self_hear(org.brain, b.speech, heard_text="ma-xyz", plan=plan, source="self")
    assert b.speech.phonemes._weights.get("ma", 0) >= w_before.get("ma", 0)


def test_social_feedback_after_vocalization(baby_store):
    b = BabyAgent(seed=13, store_path=baby_store)
    b.birth()
    b.speech_loop._last_spoke_t = __import__("time").time()
    b.speech_loop._last_spoke_text = "ba-ba"
    org = b.org
    assert org is not None
    fb = b.speech_loop.social_feedback(
        org.brain, b.speech, caregiver_text="bravo", within_seconds=5.0
    )
    assert fb.get("social") is True


def test_self_hear_api(baby_store):
    b = BabyAgent(seed=14, store_path=baby_store)
    b.birth()
    b.reflect(prompt="cosa pensi")
    r = b.self_hear(text="ciao mondo")
    assert r["feedback"].get("self_heard") is True
