"""Propagazione sparsa e scala — cervello efficiente."""

import time

from organism.brain.topology import NeuralTopology
from organism.dna.interpreter import DNAInterpreter, merge_genomes, load_yaml
from organism.runtime import OrganismRuntime
from pathlib import Path

DNA = Path(__file__).resolve().parents[1] / "organism" / "dna" / "organism_dna.yaml"


def test_sparse_propagate_only_touches_active_edges():
    brain = NeuralTopology(seed=1)
    a = brain.add_neuron("sensory", "text_semantic_encoder")
    b = brain.add_neuron("associative", "pattern_matcher")
    c = brain.add_neuron("motor", "speech_phoneme_generator")
    brain.connect(a.id, b.id, weight=0.5)
    brain.connect(b.id, c.id, weight=0.5)
    a.activation = 0.9
    brain._active.add(a.id)
    brain.propagate(steps=1)
    assert brain.neurons[b.id].activation > 0.03
    assert brain.active_count >= 1


def test_edge_keys_fast_lookup():
    brain = NeuralTopology(seed=2)
    n1 = brain.add_neuron("sensory", "t")
    n2 = brain.add_neuron("associative", "p")
    brain.connect(n1.id, n2.id)
    assert brain.has_edge(n1.id, n2.id)
    assert not brain.has_edge(n2.id, n1.id)


def test_bulk_spawn():
    brain = NeuralTopology(seed=3)
    ids = brain.add_neurons_bulk("associative", "pattern_matcher", 12_000)
    assert len(ids) == 12_000
    assert brain.neuron_count == 12_000


def test_large_tier_under_5s_propagate():
    base = load_yaml(DNA)
    overlay = {
        "scale": {
            "neuron_multiplier": 20,
            "connection_multiplier": 0.5,
            "sparse_fan_out_cap": 10,
            "energy_budget_per_tick": 50_000,
        }
    }
    dna = DNAInterpreter()
    dna.genome = merge_genomes(base, overlay)
    t0 = time.perf_counter()
    brain = dna.grow_brain(seed=7)
    birth = time.perf_counter() - t0
    assert brain.neuron_count >= 10_000
    assert brain.synapse_count >= 50_000

    for n in brain.get_neurons("sensory", "text_semantic_encoder")[:8]:
        n.activation = 0.8
        brain._active.add(n.id)
    t1 = time.perf_counter()
    brain.propagate(steps=3)
    pulse = time.perf_counter() - t1
    assert pulse < 5.0, f"propagate troppo lento: {pulse:.2f}s (birth {birth:.2f}s)"


def test_efficiency_stats():
    org = OrganismRuntime(seed=0)
    eff = org.brain.efficiency_stats()
    assert eff["sparse"] is True
    assert eff["mean_fan_out"] > 0
