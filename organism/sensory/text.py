"""Text sensory module — semantic hash embedding → spikes."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field

from organism.brain.topology import ActivePattern, NeuralTopology, Spike


@dataclass
class TextResult:
    spikes: list[Spike]
    patterns: list[ActivePattern]
    embedding: list[float]
    lexicon_hits: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)


class TextModule:
    def __init__(self, brain: NeuralTopology, lexicon: dict[str, list[str]] | None = None) -> None:
        self.brain = brain
        self.encoders = brain.get_neurons("sensory", "text_semantic_encoder")
        self.lexicon = lexicon or {}

    def perceive(self, text: str) -> TextResult:
        self._source_text = text
        t = time.time()
        embedding = self._embed(text)
        spikes: list[Spike] = []
        for enc in self.encoders:
            dim = int(enc.meta.get("dim", enc.id % len(embedding)))
            if dim < len(embedding) and embedding[dim] > 0.45:
                spikes.append(Spike(neuron_id=enc.id, timestamp=t, intensity=embedding[dim]))

        # Lexicon pattern boosts
        hits = self._lexicon_hits(text)
        if hits:
            for enc in self.encoders[: min(20, len(hits) * 5)]:
                spikes.append(Spike(neuron_id=enc.id, timestamp=t, intensity=0.75))

        self.brain.inject_spikes(spikes)
        self.brain.propagate(steps=2)
        patterns = self.brain.get_active_patterns(threshold=0.25, modality="text")

        return TextResult(
            spikes=spikes,
            patterns=patterns,
            embedding=embedding,
            lexicon_hits=hits,
            symbols=[f"TXT:hits={hits}", f"TXT:spikes={len(spikes)}"],
        )

    def _embed(self, text: str, dim: int = 128) -> list[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * dim
        for tok in tokens:
            h = hashlib.sha256(tok.encode()).digest()
            for i in range(dim):
                vec[i] += ((h[i % len(h)] / 255.0) - 0.5) * 2.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [max(0.0, min(1.0, v / norm + 0.5)) for v in vec]

    def _lexicon_hits(self, text: str) -> list[str]:
        tl = text.lower()
        return [name for name, keys in self.lexicon.items() if any(k in tl for k in keys)]
