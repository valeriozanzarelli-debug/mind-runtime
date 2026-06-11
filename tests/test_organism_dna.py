"""DNA growth and brain topology tests."""

from organism.dna.interpreter import DNAInterpreter
from organism.runtime import OrganismRuntime


def test_dna_grows_thousands_of_synapses():
    dna = DNAInterpreter()
    brain = dna.grow_brain(seed=42)
    assert brain.neuron_count >= 500
    assert brain.synapse_count >= 10_000


def test_fractal_expansion_adds_neurons():
    dna = DNAInterpreter()
    brain = dna.grow_brain(seed=1)
    fractal = [n for n in brain.neurons.values() if "fractal" in n.meta]
    assert len(fractal) > 0


def test_studio_variant_merge():
    org = OrganismRuntime.studio_assistant(seed=7)
    assert org.stats["species"] == "InkConsciousStudioAssistant"
    assert "tattoo_quote" in org.dna.pattern_lexicon()


def test_brain_visualize_json():
    org = OrganismRuntime(seed=0)
    viz = org.brain.visualize(max_nodes=10)
    assert viz["neurons"] > 0
    assert "edges_sample" in viz


def test_sleep_prunes_synapses():
    org = OrganismRuntime.studio_assistant()
    before = org.brain.synapse_count
    result = org.sleep()
    assert result["pruned_synapses"] >= 0
    assert org.brain.synapse_count <= before
