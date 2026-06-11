"""Sinapsi nuove da co-attivazione."""

from organism.brain.growth import wire_coactive
from organism.runtime import OrganismRuntime


def test_wire_coactive_grows_synapses():
    org = OrganismRuntime.baby(seed=3)
    brain = org.brain
    before = brain.synapse_count
    text_ids = [n.id for n in brain.get_neurons("sensory", "text_semantic_encoder")[:5]]
    motor_ids = [n.id for n in brain.get_neurons("motor", "speech_phoneme_generator")[:5]]
    for n in brain.neurons.values():
        if n.id in text_ids + motor_ids:
            n.activation = 0.8
    created = wire_coactive(brain, text_ids, motor_ids, max_new=4)
    assert created > 0
    assert brain.synapse_count >= before + created
