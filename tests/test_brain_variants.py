"""Profili DNA mega/giga/ultra — conteggi attesi senza grow completo in CI."""

from pathlib import Path

import yaml

from organism.dna.interpreter import DNAInterpreter, merge_genomes, load_yaml

DNA = Path(__file__).resolve().parents[1] / "organism" / "dna" / "organism_dna.yaml"
VARIANTS = Path(__file__).resolve().parents[1] / "organism" / "dna" / "variants"


def _merged(name: str) -> dict:
    return merge_genomes(load_yaml(DNA), load_yaml(VARIANTS / f"{name}.yaml"))


def test_mega_multiplier():
    g = _merged("mega")
    assert float(g["scale"]["neuron_multiplier"]) >= 1000


def test_giga_multiplier():
    g = _merged("giga")
    assert float(g["scale"]["neuron_multiplier"]) >= 2000


def test_ultra_sparse_fanout():
    g = _merged("ultra")
    assert int(g["scale"]["sparse_fan_out_cap"]) <= 5


def test_small_grow_still_fast():
    dna = DNAInterpreter()
    dna.genome = merge_genomes(load_yaml(DNA), {"scale": {"neuron_multiplier": 3.0}})
    brain = dna.grow_brain(seed=1)
    assert brain.neuron_count >= 4000
    assert brain.synapse_count >= 30_000
