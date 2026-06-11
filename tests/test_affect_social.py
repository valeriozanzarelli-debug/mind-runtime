"""Emozioni, tono sociale, correzioni."""

from pathlib import Path

import pytest

from organism.autonomous.baby_agent import BabyAgent
from organism.cognition.affect import AffectiveEngine
from organism.sensory.social_tone import analyze_social_tone, extract_correction_payload
from organism.teaching.correction import CorrectionLearner


@pytest.fixture
def baby_store(tmp_path: Path):
    return str(tmp_path / "baby.json")


def test_detect_anger():
    t = analyze_social_tone("BASTA! sei stupido!!!")
    assert t.is_angry
    assert t.valence < 0


def test_detect_correction():
    t = analyze_social_tone("no sbagliato, si dice sono organism")
    assert t.is_correction
    assert extract_correction_payload("si dice sono organism") == "sono organism"


def test_affect_updates_from_anger():
    aff = AffectiveEngine()
    tone = analyze_social_tone("sono arrabbiato con te")
    aff.update_from_social(tone)
    assert aff.state.fear > 0.2 or aff.state.shame > 0.15


def test_correction_learner(baby_store):
    b = BabyAgent(seed=30, store_path=baby_store)
    b.birth()
    b.corrections.note_baby_spoke("io sono mario")
    r = b.corrections.try_learn(
        "no sbagliato si dice sono organism",
        is_correction=True,
        dialogue_teach=b.dialogue.teach,
        phonemes=b.speech.phonemes,
        lexicon=b.composer.lexicon,
    )
    assert r.get("applied")
    assert "organism" in r.get("right", "").lower()


def test_sense_learns_from_correction(baby_store):
    b = BabyAgent(seed=31, store_path=baby_store)
    b.birth()
    for _ in range(3):
        b.teach_dialogue("come ti chiami", "mi chiamo organism")
    b.sense(text="come ti chiami")
    b._last_baby_spoke = "mi chiamo mario"
    r = b.sense(text="no sbagliato si dice mi chiamo organism")
    assert r["moment"].get("social_tone", {}).get("is_correction")
    for _ in range(3):
        b.teach_dialogue("come ti chiami", "mi chiamo organism")
    r2 = b.sense(text="come ti chiami")
    spoke = r2["moment"].get("spoke", "").lower()
    assert "organism" in spoke or r2["moment"].get("understood")


def test_brain_pulse_tick(baby_store):
    b = BabyAgent(seed=32, store_path=baby_store)
    b.birth()
    r = b.brain_pulse_tick()
    assert r.get("alive")
    assert r.get("pulses", 0) >= 1
