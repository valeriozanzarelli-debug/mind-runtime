"""Apprendimento per ripetizione — come impariamo noi umani."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TeachingTrial:
    phrase: str
    count: int = 1


class RepetitionTeacher:
    """Nessuna frase hardcoded — solo ciò che il caregiver ripete."""

    def __init__(self, consolidate_at: int = 3) -> None:
        self.consolidate_at = consolidate_at
        self._trials: dict[str, list[TeachingTrial]] = {}
        self._learned: dict[str, str] = {}

    def teach(self, stimulus_key: str, phrase: str) -> dict[str, Any]:
        phrase = phrase.strip()
        if not stimulus_key or not phrase:
            return {"learned": False, "count": 0}
        trials = self._trials.setdefault(stimulus_key, [])
        for t in trials:
            if t.phrase.lower() == phrase.lower():
                t.count += 1
                if t.count >= self.consolidate_at:
                    self._learned[stimulus_key] = phrase
                    return {"learned": True, "count": t.count, "phrase": phrase}
                return {"learned": False, "count": t.count, "phrase": phrase}
        trials.append(TeachingTrial(phrase=phrase, count=1))
        if 1 >= self.consolidate_at:
            self._learned[stimulus_key] = phrase
            return {"learned": True, "count": 1, "phrase": phrase}
        return {"learned": False, "count": 1, "phrase": phrase}

    def respond(self, stimulus_key: str) -> str | None:
        return self._learned.get(stimulus_key)

    def respond_by_text(self, text: str) -> str | None:
        """Richiama una frase appresa se le parole coincidono (es. dici «ciao»)."""
        tl = text.strip().lower()
        if not tl:
            return None
        words = {w for w in tl.split() if w}
        best_phrase: str | None = None
        best_score = 0
        for phrase in self._learned.values():
            pl = phrase.lower()
            if tl == pl or tl in pl or pl in tl:
                return phrase
            overlap = len(words & {w for w in pl.split() if w})
            if overlap > best_score:
                best_score = overlap
                best_phrase = phrase
        return best_phrase if best_score > 0 else None

    def partial(self, stimulus_key: str) -> str | None:
        """Frase più ripetuta anche se non ancora consolidata."""
        trials = self._trials.get(stimulus_key, [])
        if not trials:
            return None
        best = max(trials, key=lambda t: t.count)
        return best.phrase if best.count >= 1 else None

    def all_learned(self) -> dict[str, str]:
        return dict(self._learned)

    def to_dict(self) -> dict[str, Any]:
        return {
            "learned": self._learned,
            "trials": {
                k: [{"phrase": t.phrase, "count": t.count} for t in v]
                for k, v in self._trials.items()
            },
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._learned = dict(data.get("learned", {}))
        self._trials = {}
        for k, items in data.get("trials", {}).items():
            self._trials[k] = [TeachingTrial(phrase=i["phrase"], count=i["count"]) for i in items]
