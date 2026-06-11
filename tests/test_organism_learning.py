"""Auto-learning, persistence, ink-api bridge."""

from pathlib import Path

import pytest

from organism.integrations.ink_api import InkApiBridge, WaMessage
from organism.runtime import OrganismRuntime


@pytest.fixture
def org() -> OrganismRuntime:
    return OrganismRuntime.studio_assistant(seed=99)


def test_live_learning_strengthens_fragments(org: OrganismRuntime):
    frag = org.memory.get("lamp_compressed")
    assert frag is not None
    w0 = frag.weight
    _, _, lr = org.live({"text": "lampadina non si accende"})
    assert lr is not None
    assert lr.synapses_strengthened > 0
    assert "lamp_compressed" in lr.fragments_reinforced
    assert frag.weight > w0


def test_repeated_episodes_create_learned_fragment(org: OrganismRuntime):
    new_id = None
    for _ in range(5):
        _, _, lr = org.live({"text": "lampadina non si accende"})
        if lr and lr.new_fragment_id:
            new_id = lr.new_fragment_id
    assert new_id is not None
    assert org.memory.get(new_id) is not None


def test_save_load_roundtrip(org: OrganismRuntime, tmp_path: Path):
    for _ in range(3):
        org.live({"text": "lampadina non si accende"})
    path = tmp_path / "organism_state.json"
    org.save_state(path)
    w_before = org.brain.mean_synapse_weight()
    cycles_before = org.learner.total_cycles

    fresh = OrganismRuntime.studio_assistant(seed=99)
    assert fresh.learner.total_cycles == 0
    loaded = fresh.load_state(path)
    assert loaded["loaded"] is True
    assert loaded["synapses_updated"] > 0
    assert fresh.learner.total_cycles == cycles_before
    assert fresh.brain.mean_synapse_weight() == pytest.approx(w_before, rel=1e-5)


def test_replay_batch(org: OrganismRuntime):
    episodes = [
        {"input": {"text": "lampadina non si accende"}},
        {"input": {"text": "Ciao, vorrei prenotare per giovedì"}},
    ]
    results = org.replay(episodes)
    assert len(results) == 2
    assert results[0]["action"] == "replace_bulb"


def test_ink_api_mock_live():
    org = OrganismRuntime.studio_assistant(seed=1)
    bridge = InkApiBridge(mock=True)
    msg = WaMessage.from_mock("preventivo tattoo braccio realistico")
    reply = bridge.live_from_message(org, msg)
    assert reply.text
    assert reply.learning is not None
    assert reply.learning["synapses_strengthened"] > 0


def test_failed_outcome_reduces_weight(org: OrganismRuntime):
    frag = org.memory.get("lamp_compressed")
    w0 = frag.weight
    org.live({"text": "lampadina non si accende"}, outcome_success=False)
    assert frag.weight < w0
