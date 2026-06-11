"""Pensiero e composizione parlato emergente."""

from pathlib import Path

import pytest

from organism.autonomous.baby_agent import BabyAgent


@pytest.fixture
def baby_store(tmp_path: Path):
    return str(tmp_path / "baby.json")
from organism.cognition.thought import ThoughtEngine
from organism.motor.compose_speech import SpeechComposer
from organism.teaching.dialogue import DialogueTeacher


def test_thought_from_memory_themes(baby_store):
    b = BabyAgent(seed=7, store_path=baby_store)
    b.birth()
    for _ in range(3):
        b.teach_dialogue("ciao", "ciao, sono qui")
    org = b.org
    assert org is not None
    thought = b.thought_engine.think(
        org.brain,
        org.memory,
        b.curiosity.state,
        heard_text="ciao",
        synapses_grown=10,
    )
    assert thought.pressure > 0
    assert any("ciao" in t or "sono" in t for t in thought.themes) or thought.memory_hits >= 0


def test_compose_long_form_from_thought_only():
    comp = SpeechComposer(seed=1)
    comp.absorb(
        "nel profondo di una rete di sinapsi viveva un essere fatto solo di domande",
        boost=2.0,
    )
    comp.absorb("voce pensiero mondo sinapsi parola storia incredibile", boost=1.5)
    from organism.cognition.thought import Thought
    from organism.motor.emergent_speech import EmergentSpeechMotor
    from organism.runtime import OrganismRuntime

    org = OrganismRuntime.baby(seed=1)
    motor = EmergentSpeechMotor(org.brain, seed=1)
    thought = Thought(
        themes=["sinapsi", "voce", "mondo", "pensiero", "domande"],
        pressure=0.65,
        memory_hits=2,
    )
    out = comp.long_form(thought=thought, motor=motor)
    assert len(out.text) > 20
    assert "sinapsi" in out.text.lower() or "domande" in out.text.lower()


def test_reflect_produces_long_speech(baby_store):
    b = BabyAgent(seed=9, store_path=baby_store)
    b.birth()
    long_say = (
        "penso a quello che sento e a quello che sto imparando. "
        "ogni parola nuova cambia qualcosa dentro."
    )
    for _ in range(3):
        b.teach_dialogue("cosa pensi", long_say)
    r = b.reflect(prompt="cosa pensi")
    spoke = r["moment"]["spoke"]
    assert len(spoke) > 20
    assert "penso" in spoke.lower() or r["moment"]["from_thought"]
