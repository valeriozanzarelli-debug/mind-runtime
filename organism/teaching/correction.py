"""Apprendimento dalle correzioni del caregiver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from organism.sensory.social_tone import extract_correction_payload


@dataclass
class CorrectionEvent:
    wrong: str
    right: str
    context: str

    def to_dict(self) -> dict[str, Any]:
        return {"wrong": self.wrong[:80], "right": self.right[:80], "context": self.context[:80]}


class CorrectionLearner:
    def __init__(self) -> None:
        self._events: list[CorrectionEvent] = []
        self._last_wrong: str = ""

    def note_baby_spoke(self, text: str) -> None:
        if text.strip():
            self._last_wrong = text.strip()

    def try_learn(
        self,
        caregiver_text: str,
        *,
        is_correction: bool,
        dialogue_teach,
        phonemes,
        lexicon,
    ) -> dict[str, Any]:
        if not is_correction:
            return {"applied": False}
        right = extract_correction_payload(caregiver_text)
        if not right:
            return {"applied": False, "reason": "no_payload"}
        wrong = self._last_wrong
        when = wrong[:60] if wrong else caregiver_text[:40]
        result = dialogue_teach(when, right)
        phonemes.hear(right, boost=1.4)
        if wrong:
            from organism.teaching.phonemes import split_italian_syllables

            phonemes.penalize_mismatch(
                split_italian_syllables(wrong),
                split_italian_syllables(right),
                penalty=0.2,
            )
        lexicon.absorb(right, boost=1.5)
        ev = CorrectionEvent(wrong=wrong, right=right, context=caregiver_text)
        self._events.append(ev)
        self._last_wrong = ""
        return {
            "applied": True,
            "learned": result.get("learned", False),
            "when": when,
            "right": right,
            "event": ev.to_dict(),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "corrections": len(self._events),
            "recent": [e.to_dict() for e in self._events[-5:]],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"events": [e.to_dict() for e in self._events[-30:]]}

    def load_dict(self, data: dict[str, Any]) -> None:
        self._events = [
            CorrectionEvent(
                wrong=str(e.get("wrong", "")),
                right=str(e.get("right", "")),
                context=str(e.get("context", "")),
            )
            for e in data.get("events", [])
        ]
