"""Coscienza (workspace) e visione reale."""

from pathlib import Path

import pytest

from organism.autonomous.baby_agent import BabyAgent
from organism.cognition.workspace import GlobalWorkspace
from organism.runtime import OrganismRuntime
from organism.sensory.visual_scene import gray_to_grid, scene_signature


@pytest.fixture
def baby_store(tmp_path: Path):
    return str(tmp_path / "baby.json")


def test_scene_signature_stable():
    grid = [[100] * 8 for _ in range(8)]
    a = scene_signature(grid)
    b = scene_signature(grid)
    assert a == b
    grid[0][0] = 110
    c = scene_signature(grid)
    assert c == a  # macro-cell robusto


def test_decode_image_gray():
    flat = [128] * (16 * 16)
    grid = gray_to_grid(flat, 16, 16)
    assert len(grid) == 16
    assert grid[0][0] == 128


def test_workspace_ignition():
    org = OrganismRuntime.baby(seed=3)
    ws = GlobalWorkspace(threshold=0.2)
    from organism.cognition.thought import Thought

    for n in org.brain.get_neurons("sensory", "vision_edge_detector")[:20]:
        n.activation = 0.9
    thought = Thought(themes=["luce"], pressure=0.6)
    state = ws.cycle(org.brain, thought, sensory_symbols=["VIS:edges=0.4"], novelty=0.5)
    assert state.conscious is True
    assert state.ignition > 0.2


def test_sense_with_real_gray_grid(baby_store):
    b = BabyAgent(seed=20, store_path=baby_store)
    b.birth()
    gray = [200 if i % 2 == 0 else 40 for i in range(64 * 64)]
    r = b.sense(image_gray=gray, image_w=64, image_h=64, text="ciao")
    assert r["moment"]["consciousness"]
    assert b._last_vision_hash


def test_visual_recall_via_resolve(baby_store):
    b = BabyAgent(seed=21, store_path=baby_store)
    b.birth()
    gray = [180] * (64 * 64)
    b.sense(image_gray=gray, image_w=64, image_h=64)
    vkey = b._last_vision_hash
    from organism.drives.curiosity import stimulus_key_visual_context

    sk = stimulus_key_visual_context(vision_hash=vkey)
    for _ in range(3):
        b.teach_repetition("vedo la luce", stimulus_key=sk, image_gray=gray, image_w=64, image_h=64)
    r = b.sense(image_gray=gray, image_w=64, image_h=64)
    spoke = r["moment"].get("spoke", "")
    assert "luce" in spoke.lower() or r["moment"].get("understood")
