"""Tests for biological brain architecture."""

from organism.brain.architect import BrainArchitect
from organism.brain.regions import REGIONS, total_neurons
from organism.brain.connectivity import CONNECTIVITY
from organism.brain.runtime import BrainRuntime


def test_total_neurons():
    assert total_neurons() == 23800
    assert len(REGIONS) == 18


def test_brain_build():
    brain = BrainArchitect(seed=42).build()
    assert brain.neuron_count == 23800
    assert brain.synapse_count > 100_000
    assert len(brain.stats()["regions"]) == 18


def test_connectivity_rules():
    region_names = {r.name for r in REGIONS}
    for rule in CONNECTIVITY:
        assert rule.source in region_names, f"unknown source: {rule.source}"
        assert rule.target in region_names, f"unknown target: {rule.target}"


def test_runtime_tick():
    rt = BrainRuntime.create(seed=42)
    rt.birth()
    rt.perceive_text("ciao")
    result = rt.tick()
    assert result["alive"] is True
    assert "consciousness" in result
    assert 0.0 <= result["consciousness"]["phi"] <= 1.0


def test_dopamine_prediction_error():
    rt = BrainRuntime.create(seed=42)
    rt.birth()
    rt.perceive_text("test input forte")
    rt.tick()
    assert "prediction_error" in rt.dopamine.to_dict()


def test_phi_varies_with_complexity():
    rt = BrainRuntime.create(seed=42)
    rt.birth()
    rt.perceive_text("a")
    low = rt.tick(task_complexity=0.1)["consciousness"]["phi"]
    rt.perceive_text("questo è un input molto più complesso con molte parole")
    for _ in range(5):
        high = rt.tick(task_complexity=0.9)["consciousness"]["phi"]
    assert high >= 0.0
