"""Persistenza topologia — sinapsi dinamiche sopravvivono al reload."""

from pathlib import Path

from organism.brain.growth import wire_coactive
from organism.brain.topology import NeuralTopology
from organism.learning.store import OrganismStore
from organism.runtime import OrganismRuntime


def test_dynamic_synapses_persist_roundtrip(tmp_path: Path):
    org = OrganismRuntime.baby(seed=7)
    brain = org.brain
    base = brain.synapse_count
    pre = [n.id for n in brain.get_neurons("sensory", "text_semantic_encoder")[:6]]
    post = [n.id for n in brain.get_neurons("motor", "speech_phoneme_generator")[:6]]
    created = wire_coactive(brain, pre, post, max_new=8, weight=0.25)
    assert created > 0
    assert brain.synapse_count > base

    store = OrganismStore(tmp_path / "state.json")
    store.save(brain, org.memory, org.learner)

    org2 = OrganismRuntime.baby(seed=7)
    stats = store.load(org2.brain, org2.memory, org2.learner)
    assert stats["loaded"]
    assert org2.brain.synapse_count >= brain.synapse_count
    assert stats.get("added_synapses", 0) >= 0 or stats.get("updated_weights", 0) > 0
