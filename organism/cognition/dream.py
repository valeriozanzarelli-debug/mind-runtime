"""Sogno — riattivazione memoria durante fase delta, senza input esterno."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from mind.memory import MemoryGraph
from organism.brain.topology import NeuralTopology
from organism.cognition.neural_lexicon import NeuralLexicon


@dataclass
class DreamState:
    active: bool = False
    content: str = ""
    symbols: list[str] = field(default_factory=list)
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "content": self.content[:200],
            "symbols": self.symbols[:8],
            "depth": self.depth,
        }


class DreamEngine:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.state = DreamState()
        self._dream_count = 0

    def cycle(
        self,
        brain: NeuralTopology,
        memory: MemoryGraph,
        lexicon: NeuralLexicon | None = None,
        *,
        min_idle_s: float = 8.0,
        idle_s: float = 0.0,
    ) -> DreamState:
        if idle_s < min_idle_s:
            self.state = DreamState(active=False)
            return self.state

        frags = memory.all_fragments()
        if not frags:
            self.state = DreamState(active=True, content="", symbols=["DREAM:vuoto"])
            return self.state

        picks = self.rng.sample(frags, min(3, len(frags)))
        symbols: list[str] = []
        themes: list[str] = []
        for f in picks:
            symbols.append(f"MEM:{f.id[:12]}")
            themes.extend(f.hooks[:3])
            for w in f.title.lower().split()[:4]:
                if len(w) > 2:
                    themes.append(w)

        for n in brain.get_neurons("associative", "memory_consolidator"):
            n.activation = min(1.0, n.activation + 0.2 + self.rng.random() * 0.15)
        brain.propagate(steps=2)
        brain.reinforce_active_pathway(boost=0.02)

        pool = list(dict.fromkeys(themes))
        if not pool and lexicon is not None and lexicon.count:
            pool = lexicon.ranked(lexicon.known_words(12))[:4]
        if not pool:
            content = ""
        else:
            n = 2 + int(self.rng.random() * 4)
            chunk = [self.rng.choice(pool) for _ in range(n)]
            content = " ".join(chunk)

        self._dream_count += 1
        self.state = DreamState(
            active=True,
            content=content,
            symbols=symbols + [f"DREAM:{self._dream_count}"],
            depth=len(picks),
        )
        return self.state

    def stats(self) -> dict[str, Any]:
        return {"dreams": self._dream_count, "last": self.state.to_dict()}

    def load_dict(self, data: dict[str, Any]) -> None:
        self._dream_count = int(data.get("dreams", 0))
        last = data.get("last", {})
        if last:
            self.state = DreamState(
                active=bool(last.get("active", False)),
                content=str(last.get("content", "")),
                symbols=list(last.get("symbols", [])),
                depth=int(last.get("depth", 0)),
            )
