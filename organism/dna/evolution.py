"""Evoluzione DNA — il genoma si aggiorna da metriche, separato dalla memoria appresa."""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from organism.dna.interpreter import DNA_DIR, merge_genomes, load_yaml

EVOLVED_DIR = Path(os.environ.get("ORGANISM_DNA_DIR", "/opt/mind-runtime/data/dna"))
EVOLVED_GENESIS = EVOLVED_DIR / "evolved_genesis.yaml"
EVOLUTION_LOG = EVOLVED_DIR / "evolution_log.jsonl"


def evolved_genesis_path() -> Path:
    local = Path(__file__).resolve().parents[2] / "data" / "dna" / "evolved_genesis.yaml"
    if EVOLVED_GENESIS.exists():
        return EVOLVED_GENESIS
    return local


def genesis_variant_path() -> Path:
    """Variant da usare: evoluto se esiste, altrimenti template genesis."""
    p = evolved_genesis_path()
    if p.exists():
        return p
    return DNA_DIR / "variants" / "genesis.yaml"


class DNAEvolver:
    """Mutazioni guidate da metriche — salva solo DNA, mai lessico o dialoghi."""

    MUTABLE_PATHS: tuple[tuple[str, ...], float, float] = (
        (("scale", "neuron_multiplier"), 18.0, 32.0),
        (("scale", "connection_multiplier"), 0.55, 0.85),
        (("plasticity", "hebbian_learning", "rate"), 0.006, 0.022),
        (("plasticity", "hebbian_learning", "decay"), 0.0004, 0.002),
        (("evolution", "targets", "min_discourse_words"), 6.0, 14.0),
    )

    def __init__(self, genome: dict[str, Any] | None = None) -> None:
        self.genome = genome or load_yaml(genesis_variant_path())
        self.generation = int(self.genome.get("evolution", {}).get("generation", 0))

    @classmethod
    def load(cls) -> DNAEvolver:
        return cls(load_yaml(genesis_variant_path()))

    def _get_nested(self, keys: tuple[str, ...]) -> float:
        node: Any = self.genome
        for k in keys:
            node = node.get(k, {})
        return float(node) if not isinstance(node, dict) else 0.0

    def _set_nested(self, keys: tuple[str, ...], value: float) -> None:
        node = self.genome
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = round(value, 4)

    def evaluate_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Score 0–1 — quanto il comportamento assomiglia a pensiero/discorso umano."""
        probes = metrics.get("probe_outputs") or []
        babble_n = 0
        word_lens: list[int] = []
        for _q, spoke in probes:
            s = str(spoke).strip()
            words = s.split()
            word_lens.append(len(words))
            if len(words) < 6 or len(set(words)) < max(2, len(words) // 3):
                babble_n += 1
        babble_ratio = babble_n / max(1, len(probes))
        avg_words = sum(word_lens) / max(1, len(word_lens))
        diversity = float(metrics.get("speech_diversity", 0))
        thought_coh = float(metrics.get("thought_coherence", 0.5))
        min_words = float(
            self.genome.get("evolution", {}).get("targets", {}).get("min_discourse_words", 8)
        )
        discourse_ok = 1.0 if avg_words >= min_words else avg_words / min_words
        score = (
            0.3 * diversity
            + 0.25 * discourse_ok
            + 0.25 * thought_coh
            + 0.2 * (1.0 - babble_ratio)
        )
        return {
            "score": round(score, 3),
            "babble_ratio": round(babble_ratio, 3),
            "avg_words": round(avg_words, 1),
            "diversity": round(diversity, 3),
            "thought_coherence": round(thought_coh, 3),
        }

    def maybe_evolve(self, metrics: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
        """Se score basso o force, muta DNA e salva evolved_genesis.yaml."""
        evald = self.evaluate_metrics(metrics)
        score = evald["score"]
        targets = self.genome.setdefault("evolution", {}).setdefault("targets", {})
        max_babble = float(targets.get("max_babble_ratio", 0.25))
        improved = False
        mutations: list[dict[str, Any]] = []

        need = force or score < 0.62 or evald["babble_ratio"] > max_babble
        if need:
            direction = 1.0 if evald["babble_ratio"] > max_babble else -0.5
            if evald["avg_words"] < float(targets.get("min_discourse_words", 8)):
                direction = 1.0
            for keys, lo, hi in self.MUTABLE_PATHS:
                cur = self._get_nested(keys)
                if cur <= 0:
                    continue
                delta = cur * 0.04 * direction
                if keys[-1] == "decay":
                    delta = -delta
                new_val = max(lo, min(hi, cur + delta))
                if abs(new_val - cur) > 1e-6:
                    self._set_nested(keys, new_val)
                    mutations.append({"path": ".".join(keys), "from": cur, "to": new_val})
            if evald["avg_words"] < 8:
                targets["min_discourse_words"] = min(14, int(targets.get("min_discourse_words", 8)) + 1)
            self.generation += 1
            self.genome["evolution"]["generation"] = self.generation
            self.genome["evolution"]["last_score"] = score
            self.genome["evolution"]["last_eval"] = evald
            self.genome["genome_version"] = f"0.2.{self.generation}-genesis"
            improved = True
            self.save()

        entry = {
            "t": time.time(),
            "generation": self.generation,
            "score": score,
            "eval": evald,
            "mutations": mutations,
            "evolved": improved,
        }
        self._log(entry)
        return {"evolved": improved, "generation": self.generation, "score": score, **evald}

    def save(self) -> Path:
        EVOLVED_DIR.mkdir(parents=True, exist_ok=True)
        local = Path(__file__).resolve().parents[2] / "data" / "dna"
        local.mkdir(parents=True, exist_ok=True)
        clean = self.export_clean()
        for path in (EVOLVED_GENESIS, local / "evolved_genesis.yaml"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.dump(clean, allow_unicode=True, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        return EVOLVED_GENESIS

    def export_clean(self) -> dict[str, Any]:
        """DNA installabile — nessun dato appreso, solo capacità strutturali."""
        g = copy.deepcopy(self.genome)
        g.pop("organism", None)
        g.pop("learned", None)
        baby = g.setdefault("baby", {})
        baby["genesis"] = True
        baby["empty_knowledge"] = True
        baby.pop("research_topics", None)
        baby["research_topics"] = []
        g["pattern_lexicon"] = {}
        return g

    def _log(self, entry: dict[str, Any]) -> None:
        EVOLVED_DIR.mkdir(parents=True, exist_ok=True)
        with EVOLUTION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def stats(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "path": str(evolved_genesis_path()),
            "exists": evolved_genesis_path().exists(),
            "species": self.genome.get("species", ""),
            "version": self.genome.get("genome_version", ""),
        }
