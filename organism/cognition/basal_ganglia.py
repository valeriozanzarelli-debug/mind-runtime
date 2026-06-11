"""Gangli della base — selezione competitiva tra impulsi motori."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ActionSelection:
    impulse: str
    urgency: float
    inhibition: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "impulse": self.impulse,
            "urgency": round(self.urgency, 3),
            "inhibition": round(self.inhibition, 3),
        }


class BasalGanglia:
    """Sceglie speak / ask / silent / explore in base a drive e contesto."""

    def select(
        self,
        *,
        default: str,
        heard: str | None,
        curiosity: float,
        novelty: float,
        boredom: float,
        amygdala_inhibition: float,
        has_association: bool,
        wants_ask: bool,
    ) -> ActionSelection:
        scores: dict[str, float] = {
            "silent": 0.15 + amygdala_inhibition * 0.4,
            "explore": novelty * 0.55 + boredom * 0.35,
            "ask": (0.25 if wants_ask else 0.0) + curiosity * 0.45,
            "vocalize": boredom * 0.25 + curiosity * 0.2,
            "speak": 0.2,
        }
        if heard and heard.strip():
            scores["speak"] += 0.65
            scores["ask"] += 0.15 if wants_ask else 0.0
        if has_association:
            scores["speak"] += 0.18
        if amygdala_inhibition > 0.55:
            scores["silent"] += 0.35
            scores["speak"] *= 0.7

        impulse = max(scores, key=lambda k: scores[k])
        if scores[impulse] < 0.28 and default:
            impulse = default
        urgency = min(1.0, scores.get(impulse, 0.3))
        inhibition = min(1.0, scores["silent"] + amygdala_inhibition * 0.3)
        return ActionSelection(impulse=impulse, urgency=urgency, inhibition=inhibition)

    def stats(self) -> dict[str, Any]:
        return {"module": "basal_ganglia"}
