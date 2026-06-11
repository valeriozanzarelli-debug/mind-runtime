"""Vista umana + linguaggio emergente + coscienza autonoma."""

from pathlib import Path

import pytest

from organism.autonomous.baby_agent import BabyAgent
from organism.cognition.visual_bind import VisualBinder
from organism.cognition.workspace import GlobalWorkspace
from organism.runtime import OrganismRuntime


@pytest.fixture
def baby_store(tmp_path: Path):
    return str(tmp_path / "baby.json")


def test_visual_binder_scene_words():
    vb = VisualBinder()
    vb.bind("abc123", "vedo la luce")
    themes = vb.themes("abc123", {"luminance": 0.8, "contrast": 0.2})
    assert "vedo" in themes or "luce" in themes
    assert "VIS:bright" in themes


def test_workspace_mode_speak_vs_reflect():
    org = OrganismRuntime.baby(seed=3)
    ws = GlobalWorkspace(threshold=0.2)
    from organism.cognition.thought import Thought

    thought = Thought(themes=["a", "b"], pressure=0.5, memory_hits=1)
    for n in org.brain.get_neurons("motor", "speech_phoneme_generator")[:6]:
        n.activation = 0.7
    for n in org.brain.get_neurons("associative", "pattern_matcher")[:12]:
        n.activation = 0.65
    state = ws.cycle(
        org.brain,
        thought,
        wave_phase="think",
        has_learned_path=True,
        novelty=0.3,
    )
    assert state.conscious
    assert state.mode in ("speak", "reflect", "flow", "silent")


def test_dialogue_emerges_from_pathway_not_template(baby_store):
    b = BabyAgent(seed=4, store_path=baby_store)
    b.birth()
    for _ in range(3):
        b.teach_dialogue("come stai", "sto bene")
    m = b.sense(text="come stai")["moment"]
    assert m["understood"] or m["from_thought"]
    assert m["spoke"] or m["consciousness"].get("mode") in ("reflect", "flow", "speak")
