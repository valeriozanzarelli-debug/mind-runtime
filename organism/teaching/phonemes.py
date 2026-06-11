"""Inventario sillabe — imparato solo da ciò che sente (caregiver / microfono)."""

from __future__ import annotations

import re
import random
from typing import Any

_VOWELS = set("aeiouàèéìòù")


def split_italian_syllables(text: str) -> list[str]:
    """Sillabifica algoritmica — nessun dizionario di parole."""
    text = text.lower()
    text = re.sub(r"[^a-zàèéìòù\s]", " ", text)
    out: list[str] = []
    for word in text.split():
        word = word.strip()
        if not word:
            continue
        chunk = ""
        for i, ch in enumerate(word):
            chunk += ch
            if ch not in _VOWELS:
                continue
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt in _VOWELS:
                continue
            out.append(chunk)
            chunk = ""
        if chunk:
            if out and chunk[0] not in _VOWELS:
                out[-1] += chunk
            else:
                out.append(chunk)
    return [s for s in out if s]


class PhonemeLearner:
    """Pesi sulle sillabe udite — il balbettio campiona solo ciò che ha sentito."""

    def __init__(self) -> None:
        self._weights: dict[str, float] = {}

    @property
    def count(self) -> int:
        return len(self._weights)

    def hear(self, text: str, *, boost: float = 1.0) -> list[str]:
        heard: list[str] = []
        for syl in split_italian_syllables(text):
            self._weights[syl] = self._weights.get(syl, 0.0) + boost
            heard.append(syl)
        return heard

    def sample(self, n: int, rng: random.Random, *, curiosity: float = 0.5) -> list[str]:
        if not self._weights:
            return []
        keys = list(self._weights.keys())
        weights = [max(0.05, w * (1.0 + curiosity * 0.3)) for w in self._weights.values()]
        return [rng.choices(keys, weights=weights, k=1)[0] for _ in range(max(1, n))]

    def to_dict(self) -> dict[str, Any]:
        return {"weights": dict(self._weights)}

    def load_dict(self, data: dict[str, Any]) -> None:
        self._weights = {str(k): float(v) for k, v in data.get("weights", {}).items()}

    def reinforce_sequence(self, syllables: list[str], *, boost: float = 0.3) -> int:
        n = 0
        for syl in syllables:
            if syl in self._weights:
                self._weights[syl] = min(8.0, self._weights[syl] + boost)
                n += 1
            else:
                self._weights[syl] = boost
                n += 1
        return n

    def penalize_mismatch(
        self,
        intended: list[str],
        heard: list[str],
        *,
        penalty: float = 0.15,
    ) -> int:
        """Penalizza sillabe prodotte che non erano nel piano motorio."""
        intended_set = set(intended)
        n = 0
        for syl in heard:
            if syl not in intended_set and syl in self._weights:
                self._weights[syl] = max(0.02, self._weights[syl] - penalty)
                n += 1
        return n
