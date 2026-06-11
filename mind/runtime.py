"""MIND runtime orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

from mind.memory import MemoryGraph
from mind.pattern import PatternEngine
from mind.policy import ExperiencePolicy, is_human_interface
from mind.seed import build_seed_circuits, build_seed_memory, build_seed_patterns
from mind.sensation import SensationRegistry
from mind.types import CostLevel, Cue, CueKind, MindResult, VisualShape


class MindRuntime:
    def __init__(
        self,
        memory: MemoryGraph,
        patterns: PatternEngine,
        sensations: SensationRegistry,
        policy: ExperiencePolicy | None = None,
    ) -> None:
        self.memory = memory
        self.patterns = patterns
        self.sensations = sensations
        self.policy = policy or ExperiencePolicy()

    @classmethod
    def load_seed(cls) -> MindRuntime:
        return cls(build_seed_memory(), build_seed_patterns(), build_seed_circuits())

    @classmethod
    def blank(cls) -> MindRuntime:
        """Memoria vuota — per il neonato che impara solo da zero."""
        return cls(MemoryGraph(), PatternEngine(), SensationRegistry())

    def think(
        self,
        cue: Cue,
        *,
        cost_override: CostLevel | None = None,
        resonate_with: str | None = None,
    ) -> MindResult:
        symbols: list[str] = []
        pattern_fill: str | None = None
        human = is_human_interface(cue)

        # --- Visual pattern path ---
        if cue.kind == CueKind.VISUAL:
            shapes = _parse_visual_sequence(cue.value)
            if self.patterns.detect_gap(shapes):
                pattern_fill = self.patterns.complete_sequence(shapes)
                symbols.append(f"PATTERN:gap→fill({pattern_fill})")

        # --- Memory retrieval ---
        fragments = self.memory.retrieve(cue)
        if fragments:
            titles = [f.title for f in fragments[:3]]
            symbols.append(f"MEM:{titles}")

        # --- Sensation circuits ---
        sensation_ids = self.sensations.circuits_for_fragments(fragments)
        if sensation_ids:
            symbols.append(f"SEN:{sensation_ids}")

        # --- Resonance: K ~ X, same circuit ---
        resonate_mode = False
        if resonate_with:
            past_frags = self.memory.retrieve(Cue(kind=CueKind.TEXT, value=resonate_with))
            circuit = self.sensations.same_circuit_from_cues(cue.value, resonate_with, past_frags)
            if circuit:
                resonate_mode = True
                if circuit not in sensation_ids:
                    sensation_ids.append(circuit)
                symbols.append(f"RESONATE:{cue.value[:20]}~{resonate_with[:20]}→{circuit}")

        # --- Policy / action ---
        action = self.policy.decide(
            cue,
            fragments,
            sensation_ids=sensation_ids,
            cost_override=cost_override,
            resonate_mode=resonate_mode,
        )
        if action:
            symbols.append(f"ACT:{action.id}(u={action.utility:.2f})")

        if human:
            symbols.append("EMO:on")
        else:
            symbols.append("EMO:off")

        return MindResult(
            cue=cue,
            sensation_ids=sensation_ids,
            fragments=fragments,
            pattern_fill=pattern_fill,
            action=action,
            explanation_symbols=symbols,
            human_emotion_active=human,
        )

    def think_json(self, cue: Cue, **kwargs) -> str:
        return json.dumps(self.think(cue, **kwargs).to_dict(), ensure_ascii=False, indent=2)


def _parse_visual_sequence(value: str) -> list[VisualShape]:
    """
    Format: 'quadrato+cerchio,triangolo+cerchio,rettangolo+'
    or 'quadrato+cerchio;triangolo+cerchio;rettangolo'
    """
    sep = ";" if ";" in value else ","
    shapes: list[VisualShape] = []
    for part in value.split(sep):
        part = part.strip()
        if not part:
            continue
        if "+" in part:
            outer, inner = part.split("+", 1)
            inner = inner.strip() or None
        else:
            outer, inner = part, None
        shapes.append(VisualShape(outer.strip(), inner))
    return shapes


def demo_scenarios() -> list[tuple[str, Cue, dict]]:
    """Named scenarios for quick testing."""
    return [
        (
            "Puzzle forme — cerchio mancante",
            Cue(kind=CueKind.VISUAL, value="quadrato+cerchio,triangolo+cerchio,rettangolo+"),
            {},
        ),
        (
            "Lampadina — costo basso",
            Cue(kind=CueKind.TEXT, value="la lampadina non si accende"),
            {},
        ),
        (
            "Lampadina — costo alto (test prima)",
            Cue(kind=CueKind.TEXT, value="la lampadina non si accende"),
            {"cost_override": CostLevel.HIGH},
        ),
        (
            "Matrimonio fratello — link nascosto",
            Cue(kind=CueKind.TEXT, value="matrimonio di mio fratello"),
            {},
        ),
        (
            "Cliente WA nuovo ma risonanza marzo",
            Cue(
                kind=CueKind.TEXT,
                value="cliente whatsapp diffidente chiede preventivo braccio",
                meta={"human": True},
            ),
            {"resonate_with": "cliente marzo diffidente whatsapp"},
        ),
    ]
