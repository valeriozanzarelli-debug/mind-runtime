"""Grafo causale esplicito — regole apprese, non correlazione statistica."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from organism.teaching.dialogue import normalize_text

_TOKEN = re.compile(r"[a-zàèéìòù']+")


@dataclass
class CausalLink:
    cause: str
    effect: str
    when: str = ""
    strength: float = 1.0


class CausalGraph:
    """Memoria procedurale: se X allora Y (Piaget: assimilazione di regole)."""

    def __init__(self) -> None:
        self._links: list[CausalLink] = []

    def teach(self, when: str, effect: str, *, cause: str = "") -> None:
        w = normalize_text(when)
        e = effect.strip()
        if not w or not e:
            return
        c = cause.strip().lower() or _extract_cause(w)
        self._links = [lk for lk in self._links if normalize_text(lk.when) != w]
        self._links.append(CausalLink(cause=c, effect=e, when=w, strength=1.0))

    def reinforce(self, when: str, *, delta: float = 0.15) -> None:
        w = normalize_text(when)
        for lk in self._links:
            if normalize_text(lk.when) == w:
                lk.strength = min(3.0, lk.strength + delta)

    def lookup(self, heard: str) -> CausalLink | None:
        hn = normalize_text(heard)
        if not hn:
            return None
        hw = _tokens(hn)
        best: tuple[float, CausalLink] | None = None
        for lk in self._links:
            tw = _tokens(normalize_text(lk.when))
            if not tw:
                continue
            overlap = len(hw & tw) / max(1, len(tw))
            if hn == normalize_text(lk.when):
                overlap = 1.0
            if overlap < 0.45:
                continue
            score = overlap * lk.strength
            if best is None or score > best[0]:
                best = (score, lk)
        return best[1] if best else None

    def seed_from_pairs(self, pairs: list[tuple[str, str]]) -> int:
        n = 0
        for when, say in pairs:
            self.teach(when, say)
            n += 1
        return n

    def to_dict(self) -> dict[str, Any]:
        return {
            "links": [
                {
                    "when": lk.when,
                    "cause": lk.cause,
                    "effect": lk.effect,
                    "strength": lk.strength,
                }
                for lk in self._links
            ]
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._links = []
        for row in data.get("links", []):
            self._links.append(
                CausalLink(
                    cause=str(row.get("cause", "")),
                    effect=str(row.get("effect", "")),
                    when=str(row.get("when", "")),
                    strength=float(row.get("strength", 1.0)),
                )
            )


def _tokens(text: str) -> set[str]:
    return {w for w in _TOKEN.findall(text.lower()) if len(w) > 2}


def _extract_cause(when: str) -> str:
    for cue in ("perché", "perche", "se", "quando"):
        if cue in when:
            parts = when.split(cue, 1)
            if len(parts) > 1:
                return parts[1].strip()[:40]
    return when[:40]
