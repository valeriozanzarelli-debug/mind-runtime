"""Pensiero — emerge da attivazione neurale e memoria, non da template."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mind.memory import MemoryGraph
from mind.types import Cue, CueKind
from organism.brain.topology import NeuralTopology
from organism.drives.curiosity import CuriosityState


@dataclass
class Thought:
    symbols: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    pressure: float = 0.0
    memory_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": self.symbols,
            "themes": self.themes[:12],
            "pressure": round(self.pressure, 3),
            "memory_hits": self.memory_hits,
        }


class ThoughtEngine:
    """Costruisce uno stato mentale osservabile prima di parlare."""

    def think(
        self,
        brain: NeuralTopology,
        memory: MemoryGraph,
        curiosity: CuriosityState,
        *,
        heard_text: str = "",
        synapses_grown: int = 0,
        workspace_broadcast: list[str] | None = None,
        visual_cue: str = "",
        visual_themes: list[str] | None = None,
    ) -> Thought:
        layers = brain.layer_activation_summary()
        motor = float(layers.get("motor", 0.0))
        sensory = float(layers.get("sensory", 0.0))
        assoc = float(layers.get("associative", 0.0))

        symbols = [
            f"BRAIN:sensory={sensory:.2f}",
            f"BRAIN:assoc={assoc:.2f}",
            f"BRAIN:motor={motor:.2f}",
            f"BRAIN:tick={int(brain.tick)}",
            f"BRAIN:synapses+={synapses_grown}",
            f"CURIOSITY:{curiosity.level:.2f}",
            f"NOVELTY:{curiosity.novelty:.2f}",
        ]

        themes: list[str] = []
        hits = 0
        heard_words = {
            w for w in heard_text.lower().split() if len(w) > 2
        } if heard_text.strip() else set()
        if heard_text.strip():
            cue = Cue(kind=CueKind.TEXT, value=heard_text)
            frags = memory.retrieve(cue, limit=6)
            hits = len(frags)
            for f in frags:
                symbols.append(f"MEM:{f.id[:20]}")
                for h in f.hooks[:4]:
                    if not h:
                        continue
                    hl = h.lower()
                    if heard_words and hl not in heard_words and not any(hw in hl or hl in hw for hw in heard_words):
                        continue
                    if h not in themes:
                        themes.append(h)
                for w in f.title.lower().split()[:5]:
                    if len(w) > 2 and w not in themes:
                        if heard_words and w not in heard_words:
                            continue
                        themes.append(w)

        if visual_cue:
            symbols.append(f"VIS:sig={visual_cue[:16]}")
        for vt in visual_themes or []:
            symbols.append(f"VIS:{vt[:20]}" if not vt.startswith("VIS:") else vt)
            tag = vt.replace("VIS:", "")
            if len(tag) > 2 and tag not in themes and not tag.startswith("sig="):
                themes.append(tag)

        for b in workspace_broadcast or []:
            tag = b.replace("VIS:", "").replace("MEM:", "")[:24]
            if tag and tag not in themes:
                themes.append(tag)

        if not themes and curiosity.level > 0.4 and not heard_text.strip():
            for f in memory.all_fragments()[-4:]:
                for h in f.hooks[:2]:
                    if h not in themes:
                        themes.append(h)

        question_boost = 0.0
        if heard_text.strip():
            ht = heard_text.strip().lower()
            if ht.endswith("?") or any(
                ht.startswith(w) for w in ("chi ", "come ", "cosa ", "dove ", "quando ", "perch")
            ):
                question_boost = 0.22
                symbols.append("QUESTION:yes")

        pressure = min(
            1.0,
            0.25 * curiosity.level
            + 0.2 * motor
            + 0.15 * sensory
            + 0.1 * min(1.0, hits / 4)
            + 0.05 * min(1.0, synapses_grown / 500)
            + 0.08 * (0.15 if workspace_broadcast else 0.0)
            + question_boost,
        )

        return Thought(symbols=symbols, themes=themes, pressure=pressure, memory_hits=hits)
