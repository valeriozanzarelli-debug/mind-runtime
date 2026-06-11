"""Dialogo — quando senti X, rispondi Y (appreso per ripetizione, non template)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

Kind = Literal["speech", "code"]


@dataclass
class PairTrial:
    when: str
    say: str
    kind: Kind
    count: int = 1


def normalize_text(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


class DialogueTeacher:
    """Coppie udito→risposta — il bambino capisce cosa rispondere a cosa."""

    def __init__(self, consolidate_at: int = 3) -> None:
        self.consolidate_at = consolidate_at
        self._trials: list[PairTrial] = []
        self._learned: list[tuple[str, str, Kind]] = []

    def teach(self, when: str, say: str, *, kind: Kind = "speech") -> dict[str, Any]:
        when_n = normalize_text(when)
        say = say.strip()
        if not when_n or not say:
            return {"learned": False, "count": 0}
        for t in self._trials:
            if t.when == when_n and t.say == say and t.kind == kind:
                t.count += 1
                if t.count >= self.consolidate_at:
                    self._consolidate(when_n, say, kind)
                    return {"learned": True, "count": t.count, "when": when, "say": say, "kind": kind}
                return {"learned": False, "count": t.count, "when": when, "say": say, "kind": kind}
        self._trials.append(PairTrial(when=when_n, say=say, kind=kind, count=1))
        if 1 >= self.consolidate_at:
            self._consolidate(when_n, say, kind)
            return {"learned": True, "count": 1, "when": when, "say": say, "kind": kind}
        return {"learned": False, "count": 1, "when": when, "say": say, "kind": kind}

    def _consolidate(self, when_n: str, say: str, kind: Kind) -> None:
        key = (when_n, kind)
        self._learned = [(w, s, k) for w, s, k in self._learned if not (w == when_n and k == kind)]
        self._learned.append((when_n, say, kind))

    def best_match(self, heard: str, *, min_score: int = 200) -> tuple[str, str, Kind, int] | None:
        hn = normalize_text(heard)
        if not hn:
            return None
        best: tuple[str, str, Kind, int] | None = None
        for when_n, say, kind in self._learned:
            score = _match_score(hn, when_n)
            if score < min_score:
                continue
            if best is None or score > best[3] or (score == best[3] and len(when_n) > len(best[0])):
                best = (when_n, say, kind, score)
        return best

    def respond(self, heard: str, *, min_score: int = 200) -> tuple[str | None, Kind | None]:
        best = self.best_match(heard, min_score=min_score)
        if not best:
            return None, None
        when_n, say, kind, _ = best
        hw = _content_words(normalize_text(heard))
        tw = _content_words(when_n)
        if tw and not (tw & hw):
            return None, None
        return say, kind

    def all_pairs(self) -> list[dict[str, Any]]:
        return [{"when": w, "say": s, "kind": k} for w, s, k in self._learned]

    def to_dict(self) -> dict[str, Any]:
        return {
            "learned": [{"when": w, "say": s, "kind": k} for w, s, k in self._learned],
            "trials": [
                {"when": t.when, "say": t.say, "kind": t.kind, "count": t.count} for t in self._trials
            ],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._learned = [
            (i["when"], i["say"], i.get("kind", "speech")) for i in data.get("learned", [])
        ]
        self._trials = [
            PairTrial(
                when=i["when"],
                say=i["say"],
                kind=i.get("kind", "speech"),
                count=int(i.get("count", 1)),
            )
            for i in data.get("trials", [])
        ]


_STOP = frozenset(
    {
        "che", "chi", "come", "cosa", "dove", "quando", "perché", "perche", "quale",
        "il", "la", "lo", "un", "una", "di", "da", "in", "con", "per", "non", "mi", "ti",
        "sei", "sono", "è", "era", "del", "della", "questo", "questa", "the", "un",
    }
)


def _content_words(text: str) -> set[str]:
    return {w for w in text.split() if len(w) > 2 and w not in _STOP}


def _match_score(heard: str, trigger: str) -> int:
    if heard == trigger:
        return 1000 + len(trigger)
    if heard.startswith(trigger + " ") or heard.endswith(" " + trigger):
        return 500 + len(trigger)
    if trigger in heard.split():
        return 400 + len(trigger)
    hw = _content_words(heard)
    tw = _content_words(trigger)
    if not tw:
        return 0
    overlap = len(tw & hw)
    if overlap == len(tw):
        return 300 + overlap * 30 + len(trigger)
    if overlap == 0:
        return 0
    return 40 + overlap * 8
