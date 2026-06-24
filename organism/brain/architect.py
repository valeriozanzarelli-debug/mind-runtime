"""Brain architect — builds the 22k biological brain from DNA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from organism.brain.connectivity import CONNECTIVITY
from organism.brain.regions import REGIONS, total_neurons
from organism.brain.topology import NeuralTopology


class BrainArchitect:
    """Constructs the full biological brain topology from region specs and connectivity rules."""

    def __init__(self, seed: int = 42, dna_path: Path | None = None) -> None:
        self.seed = seed
        self.dna_path = dna_path or Path(__file__).parent.parent / "dna" / "biological_22k.yaml"
        self._dna: dict[str, Any] = {}

    def load_dna(self) -> dict[str, Any]:
        if self.dna_path.exists():
            self._dna = yaml.safe_load(self.dna_path.read_text(encoding="utf-8")) or {}
        else:
            self._dna = self._default_dna()
        return self._dna

    def build(self) -> NeuralTopology:
        dna = self.load_dna()
        brain = NeuralTopology(seed=self.seed)
        plasticity_cfg = dna.get("plasticity", {})
        brain.set_plasticity(plasticity_cfg)

        region_counts = dna.get("neuron_counts", {})
        for spec in REGIONS:
            count = region_counts.get(spec.name, spec.neuron_count)
            brain.add_neurons_bulk(spec.system.value, spec.name, count)

        conn_rules = dna.get("connectivity", None)
        rules = CONNECTIVITY if conn_rules is None else self._parse_conn_rules(conn_rules)

        total_synapses = 0
        for rule in rules:
            n = brain.connect_regions(
                rule.source,
                rule.target,
                connections_per_neuron=rule.connections_per_neuron,
                weight_init=rule.weight_init,
                pathway=rule.pathway,
                plastic=rule.plastic,
                dopamine_modulated=rule.dopamine_modulated,
            )
            total_synapses += n
            if rule.bidirectional:
                n2 = brain.connect_regions(
                    rule.target,
                    rule.source,
                    connections_per_neuron=rule.connections_per_neuron,
                    weight_init=rule.weight_init,
                    pathway=rule.pathway + "_rev",
                    plastic=rule.plastic,
                    dopamine_modulated=rule.dopamine_modulated,
                )
                total_synapses += n2

        brain.energy_budget = dna.get("energy_budget", 0)
        return brain

    def summary(self) -> dict[str, Any]:
        dna = self.load_dna()
        return {
            "genome_version": dna.get("genome_version", "1.0.0"),
            "species": dna.get("species", "InkBiologicalBrain"),
            "total_neurons": total_neurons(),
            "regions": len(REGIONS),
            "connectivity_rules": len(CONNECTIVITY),
            "systems": list({r.system.value for r in REGIONS}),
        }

    def _default_dna(self) -> dict[str, Any]:
        return {
            "genome_version": "1.0.0",
            "species": "InkBiologicalBrain",
            "neuron_counts": {r.name: r.neuron_count for r in REGIONS},
        }

    def _parse_conn_rules(self, raw: list[dict]) -> list:
        from organism.brain.connectivity import ConnectionRule

        return [ConnectionRule(**r) for r in raw]
