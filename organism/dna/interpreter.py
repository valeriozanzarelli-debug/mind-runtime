"""DNA interpreter — compressed genome → neural topology at runtime."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from organism.brain.topology import NeuralTopology

DNA_DIR = Path(__file__).parent
BASE_GENOME = DNA_DIR / "organism_dna.yaml"


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_genomes(base: dict, overlay: dict) -> dict:
    """Deep merge overlay onto base (overlay wins on leaves)."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_genomes(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class DNAInterpreter:
    def __init__(self, genome_path: str | Path | None = None, variant_path: str | Path | None = None):
        base = load_yaml(BASE_GENOME)
        if variant_path:
            overlay = load_yaml(variant_path)
            self.genome = merge_genomes(base, overlay)
        elif genome_path:
            path = Path(genome_path)
            if path.name != "organism_dna.yaml" and (DNA_DIR / "organism_dna.yaml").exists():
                self.genome = merge_genomes(base, load_yaml(path))
            else:
                self.genome = load_yaml(path)
        else:
            self.genome = base

    def grow_brain(self, seed: int = 42) -> NeuralTopology:
        brain = NeuralTopology(seed=seed)
        scale = self.genome.get("scale", {})
        mult = float(scale.get("neuron_multiplier", 1.0))
        if os.environ.get("ORGANISM_NEURON_MULTIPLIER"):
            mult = float(os.environ["ORGANISM_NEURON_MULTIPLIER"])
        conn_mult = float(scale.get("connection_multiplier", 1.0))
        if os.environ.get("ORGANISM_CONNECTION_MULTIPLIER"):
            conn_mult = float(os.environ["ORGANISM_CONNECTION_MULTIPLIER"])
        fan_cap = int(scale.get("sparse_fan_out_cap", 0))
        brain.energy_budget = int(scale.get("energy_budget_per_tick", 0))
        counts: dict[str, int] = self.genome.get("neuron_counts", {})

        # 1. Base neuron types
        for layer, subtypes in self.genome.get("neuron_types", {}).items():
            for subtype in subtypes:
                n = max(1, int(counts.get(subtype, 50) * mult))
                self._spawn_neurons(brain, layer, subtype, n)

        # 2. Fractal expansion (recursive associative subnets)
        fractal = self.genome.get("fractal_expansion", {})
        if fractal.get("enabled"):
            self._fractal_expand(brain, fractal, mult)

        # 3. Growth rules → synapses
        for rule_name, rule in self.genome.get("growth_rules", {}).items():
            if rule_name == "fractal_expansion":
                continue
            cpn = int(rule.get("connections_per_neuron", 30) * conn_mult)
            if fan_cap > 0:
                cpn = min(cpn, fan_cap)
            brain.connect_layers(
                rule.get("source_layer", rule_name.split("_to_")[0]),
                rule.get("target_layer", rule_name.split("_to_")[-1]),
                source_subtype=rule.get("source_subtype"),
                target_subtype=rule.get("target_subtype"),
                connections_per_neuron=cpn,
                weight_init=rule.get("weight_init", "xavier_uniform"),
                pruning_threshold=float(rule.get("pruning_threshold", 0.0)),
            )

        # 4. Plasticity
        brain.set_plasticity(self.genome.get("plasticity", {}))
        return brain

    def pattern_lexicon(self) -> dict[str, list[str]]:
        return self.genome.get("pattern_lexicon", {})

    def motor_defaults(self) -> dict[str, Any]:
        return self.genome.get("motor_defaults", {})

    def pruning_config(self) -> dict[str, Any]:
        return self.genome.get("pruning", {})

    def _spawn_neurons(self, brain: NeuralTopology, layer: str, subtype: str, count: int) -> None:
        if count <= 0:
            return
        if count >= 5000:
            brain.add_neurons_bulk(layer, subtype, count, meta_fn=self._meta_fn(subtype))
            return
        for i in range(count):
            brain.add_neuron(layer, subtype, self._neuron_meta(subtype, i, count))

    def _meta_fn(self, subtype: str):
        def factory(i: int, count: int) -> dict[str, Any]:
            return self._neuron_meta(subtype, i, count)

        return factory

    def _neuron_meta(self, subtype: str, i: int, count: int) -> dict[str, Any]:
        meta: dict[str, Any] = {"index": i}
        if subtype == "vision_edge_detector":
            side = max(1, int(count**0.5))
            meta["receptive_field"] = [i % side, i // side]
        elif subtype == "audio_frequency_analyzer":
            low = 20 * (1.15 ** min(i, 120))
            meta["frequency_band"] = [low, low * 1.15]
            meta["threshold"] = 0.1
        elif subtype == "text_semantic_encoder":
            meta["dim"] = i % 4096
        return meta

    def _fractal_expand(self, brain: NeuralTopology, cfg: dict, mult: float) -> None:
        depth = int(cfg.get("depth", 3))
        factor = int(cfg.get("branching_factor", 3))
        per_leaf = max(1, int(cfg.get("neurons_per_leaf", 10) * mult))
        layer = cfg.get("layer", "associative")
        template = cfg.get("template_subtype", "pattern_matcher")

        def expand(level: int, parent_tag: str) -> None:
            if level >= depth:
                for j in range(per_leaf):
                    brain.add_neuron(layer, template, {"fractal": f"{parent_tag}.{j}", "depth": level})
                return
            for b in range(factor):
                expand(level + 1, f"{parent_tag}_{b}")

        expand(0, "root")
