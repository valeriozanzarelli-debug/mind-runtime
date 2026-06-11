"""Bridge sensory patterns → MIND Cue + enrich MindResult with brain state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mind.runtime import MindRuntime
from mind.types import CostLevel, Cue, CueKind, MindResult
from organism.brain.topology import ActivePattern


@dataclass
class SensoryBundle:
    vision: Any | None = None
    audio: Any | None = None
    text: Any | None = None
    _raw_text: str = ""
    _raw_shapes: str = ""

    def all_patterns(self) -> list[ActivePattern]:
        out: list[ActivePattern] = []
        for mod in (self.vision, self.audio, self.text):
            if mod is not None and hasattr(mod, "patterns"):
                out.extend(mod.patterns)
        return out

    def all_symbols(self) -> list[str]:
        syms: list[str] = []
        for mod in (self.vision, self.audio, self.text):
            if mod is not None and hasattr(mod, "symbols"):
                syms.extend(mod.symbols)
        return syms

    def lexicon_hits(self) -> list[str]:
        if self.text is not None and hasattr(self.text, "lexicon_hits"):
            return list(self.text.lexicon_hits)
        return []


@dataclass
class OrganismThought:
    mind_result: MindResult
    sensory: SensoryBundle
    fused_cue: Cue
    brain_patterns: list[ActivePattern] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)


class MindBridge:
    def __init__(self, mind: MindRuntime | None = None) -> None:
        self.mind = mind or MindRuntime.load_seed()

    def think(
        self,
        sensory: SensoryBundle,
        *,
        cost_override: CostLevel | None = None,
        resonate_with: str | None = None,
    ) -> OrganismThought:
        cue = self._fuse_cue(sensory)
        mind_result = self.mind.think(cue, cost_override=cost_override, resonate_with=resonate_with)
        patterns = sensory.all_patterns()
        symbols = sensory.all_symbols() + mind_result.explanation_symbols
        if patterns:
            symbols.append(f"BRAIN:patterns={len(patterns)}")
        return OrganismThought(
            mind_result=mind_result,
            sensory=sensory,
            fused_cue=cue,
            brain_patterns=patterns,
            symbols=symbols,
        )

    def _fuse_cue(self, sensory: SensoryBundle) -> Cue:
        if sensory._raw_shapes:
            return Cue(kind=CueKind.VISUAL, value=sensory._raw_shapes)
        if sensory._raw_text:
            t = sensory._raw_text.strip()
            human = len(t) >= 2 and any(c.isalpha() for c in t)
            return Cue(kind=CueKind.TEXT, value=sensory._raw_text, meta={"human": human})
        if sensory.text is not None:
            src = getattr(sensory.text, "_source_text", "")
            if src:
                return Cue(kind=CueKind.TEXT, value=src, meta={"human": True})
        return Cue(kind=CueKind.TEXT, value="unknown")
