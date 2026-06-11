"""Core data types for the MIND runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CueKind(str, Enum):
    TEXT = "text"
    VISUAL = "visual"
    AUDIO = "audio"
    FUNCTION = "function"


class CostLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Cue:
    kind: CueKind
    value: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualShape:
    """outer + optional inner symbol (e.g. square + circle)."""

    outer: str
    inner: str | None = None

    def key(self) -> str:
        inner = self.inner or "∅"
        return f"{self.outer}+{inner}"


@dataclass
class HiddenDetail:
    key: str
    retrieve_only_if_cue_contains: list[str]


@dataclass
class Outcome:
    action: str
    success: bool
    cost: CostLevel = CostLevel.LOW


@dataclass
class Fragment:
    id: str
    title: str
    weight: float
    sensation_id: str
    hooks: list[str]
    links: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    hidden_details: list[HiddenDetail] = field(default_factory=list)
    outcomes: list[Outcome] = field(default_factory=list)

    def matches_hook(self, token: str) -> float:
        token_l = token.lower()
        score = 0.0
        for hook in self.hooks:
            h = hook.lower()
            if h == token_l:
                score = max(score, 1.0)
            elif h in token_l or token_l in h:
                score = max(score, 0.65)
        for fn in self.functions:
            fn_l = fn.lower()
            if fn_l in token_l or token_l in fn_l:
                score = max(score, 0.5)
        return score


@dataclass
class Circuit:
    id: str
    label: str
    fragment_ids: list[str] = field(default_factory=list)


@dataclass
class Schema:
    """Inferred visual pattern: container shapes tend to have the same inner."""

    id: str
    inner_invariant: str
    examples: list[VisualShape]
    confidence: float

    def predict_inner(self, outer: str) -> str | None:
        if self.confidence < 0.5:
            return None
        known_outers = {e.outer for e in self.examples}
        if known_outers and outer not in known_outers:
            # generalize: new outer still gets invariant if we saw enough examples
            if len(self.examples) >= 2:
                return self.inner_invariant
        if any(e.outer == outer for e in self.examples):
            return self.inner_invariant
        if len(self.examples) >= 2:
            return self.inner_invariant
        return None


@dataclass
class Action:
    id: str
    label: str
    cost: CostLevel = CostLevel.LOW
    utility: float = 0.0
    reason: str = ""


@dataclass
class MindResult:
    cue: Cue
    sensation_ids: list[str]
    fragments: list[Fragment]
    pattern_fill: str | None
    action: Action | None
    explanation_symbols: list[str]
    human_emotion_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue": {"kind": self.cue.kind.value, "value": self.cue.value},
            "sensations": self.sensation_ids,
            "fragments": [f.title for f in self.fragments],
            "pattern_fill": self.pattern_fill,
            "action": None
            if self.action is None
            else {
                "id": self.action.id,
                "label": self.action.label,
                "cost": self.action.cost.value,
                "utility": round(self.action.utility, 3),
                "reason": self.action.reason,
            },
            "symbols": self.explanation_symbols,
            "emotion_module": self.human_emotion_active,
        }
