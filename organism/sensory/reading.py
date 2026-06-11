"""Lettura — testo come flusso sensoriale; impara a leggere da solo."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from organism.brain.topology import NeuralTopology, Spike


@dataclass
class ReadingResult:
    symbols: list[str] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    chars: int = 0

    def to_dict(self) -> dict:
        return {"symbols": self.symbols, "words": self.words[:20], "chars": self.chars}


class ReadingChannel:
    """Area lettura — attiva encoder testuali come sequenza (non LLM)."""

    def __init__(self, brain: NeuralTopology) -> None:
        self.brain = brain
        self.encoders = brain.get_neurons("sensory", "text_semantic_encoder")

    def perceive(self, text: str) -> ReadingResult:
        t = time.time()
        words = [w for w in re.findall(r"[a-zàèéìòùA-ZÀÈÉÌÒÙ']+", text) if len(w) > 1]
        spikes: list[Spike] = []
        for i, w in enumerate(words[:64]):
            if i < len(self.encoders):
                enc = self.encoders[i]
                dim = int(enc.meta.get("dim", i % 4096))
                intensity = min(1.0, 0.35 + len(w) * 0.04)
                spikes.append(Spike(neuron_id=enc.id, timestamp=t, intensity=intensity))
        self.brain.inject_spikes(spikes)
        self.brain.propagate(steps=2)
        symbols = [f"READ:words={len(words)}", f"READ:chars={len(text)}"]
        for w in words[:6]:
            symbols.append(f"WORD:{w.lower()[:16]}")
        return ReadingResult(symbols=symbols, words=words, chars=len(text))
