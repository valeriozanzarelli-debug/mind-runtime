"""Syntax Planner — transizioni bigramma Hebbiane per assemblare frasi italiane."""

from __future__ import annotations

import re
import time
from typing import Any

_TOKEN_RE = re.compile(r"[a-zàèéìòù']+")
_STOP = frozenset({"il", "la", "lo", "le", "un", "una", "di", "da", "in", "su", "per", "con"})


def _tokens(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if len(w) >= 1]


class SyntaxPlanner:
    """Rinforza transizioni osservate in frasi grammaticali; inferenza greedy."""

    def __init__(self, *, boost: float = 0.05, decay: float = 0.0003) -> None:
        self.boost = boost
        self.decay = decay
        self._transition: dict[str, dict[str, float]] = {}
        self._word_weight: dict[str, float] = {}
        self._starts: dict[str, float] = {}
        self._trained = 0

    def train_sentence(self, sentence: str, *, boost: float | None = None) -> int:
        """Rinforza bigrammi in una frase grammaticale."""
        words = _tokens(sentence)
        if len(words) < 2:
            return 0
        b = boost if boost is not None else self.boost
        n = 0
        self._starts[words[0]] = min(3.0, self._starts.get(words[0], 0.0) + b)
        self._word_weight[words[0]] = min(2.0, self._word_weight.get(words[0], 0.0) + b * 0.5)
        for i in range(len(words) - 1):
            a, c = words[i], words[i + 1]
            bucket = self._transition.setdefault(a, {})
            bucket[c] = min(3.0, bucket.get(c, 0.0) + b)
            self._word_weight[c] = min(2.0, self._word_weight.get(c, 0.0) + b * 0.4)
            n += 1
        self._trained += 1
        return n

    def train_curriculum(self, sentences: list[str], *, repeats: int = 3) -> dict[str, int]:
        total = 0
        for _ in range(max(1, repeats)):
            for s in sentences:
                total += self.train_sentence(s)
        return {"sentences": len(sentences), "bigrams": total, "repeats": repeats}

    def word_weight(self, word: str) -> float:
        return self._word_weight.get(word.lower(), 0.0)

    def is_automatic(self, word: str) -> bool:
        return self.word_weight(word) >= 0.8

    def transition_weight(self, prev: str, nxt: str) -> float:
        return self._transition.get(prev.lower(), {}).get(nxt.lower(), 0.0)

    def greedy_assembly(
        self,
        themes: list[str],
        *,
        max_words: int = 15,
        min_words: int = 6,
        start_hint: str = "",
    ) -> list[str]:
        """Assembla temi in sequenza max-weight path."""
        pool = [t.lower().strip() for t in themes if t and t.isalpha()]
        pool = list(dict.fromkeys(pool))
        if not pool:
            return []

        start = start_hint.lower() if start_hint else ""
        if not start or start not in pool:
            if "io" in pool:
                start = "io"
            elif self._starts:
                scored = sorted(
                    ((self._starts.get(w, 0.0) + self.word_weight(w), w) for w in pool),
                    reverse=True,
                )
                start = scored[0][1] if scored else pool[0]
            else:
                start = pool[0]

        path = [start]
        used = {start}
        remaining = [w for w in pool if w != start]

        while len(path) < max_words:
            prev = path[-1]
            followers = self._transition.get(prev, {})
            best_w = ""
            best_score = -1.0
            for w in remaining:
                tw = followers.get(w, 0.0)
                ww = self.word_weight(w)
                score = tw * 2.0 + ww + (0.3 if w in pool else 0.0)
                if score > best_score:
                    best_score = score
                    best_w = w
            if best_w and best_score > 0.05:
                path.append(best_w)
                used.add(best_w)
                remaining = [w for w in remaining if w != best_w]
            elif remaining:
                path.append(remaining.pop(0))
            else:
                break

        # Inserisci filler grammaticali leggeri se path troppo corto
        if len(path) < min_words and remaining:
            path.extend(remaining[: min_words - len(path)])

        return path[:max_words]

    def assemble_sentence(
        self,
        themes: list[str],
        *,
        max_words: int = 15,
        min_words: int = 6,
        start_hint: str = "",
        punct: str = ".",
    ) -> str:
        words = self.greedy_assembly(
            themes, max_words=max_words, min_words=min_words, start_hint=start_hint
        )
        if not words:
            return ""
        text = " ".join(words)
        if punct and text[-1] not in ".?!":
            text += punct
        return text[0].upper() + text[1:]

    def decay_tick(self) -> None:
        """Decadimento lento transizioni non usate."""
        for a in list(self._transition.keys()):
            for b in list(self._transition[a].keys()):
                self._transition[a][b] = max(0.0, self._transition[a][b] - self.decay)
                if self._transition[a][b] <= 0:
                    del self._transition[a][b]
            if not self._transition[a]:
                del self._transition[a]
        for w in list(self._word_weight.keys()):
            self._word_weight[w] = max(0.0, self._word_weight[w] - self.decay * 0.5)

    def stats(self) -> dict[str, Any]:
        edges = sum(len(v) for v in self._transition.values())
        return {
            "trained": self._trained,
            "words": len(self._word_weight),
            "transitions": edges,
            "starts": len(self._starts),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "boost": self.boost,
            "decay": self.decay,
            "transition": {k: dict(v) for k, v in self._transition.items()},
            "word_weight": dict(self._word_weight),
            "starts": dict(self._starts),
            "trained": self._trained,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self.boost = float(data.get("boost", self.boost))
        self.decay = float(data.get("decay", self.decay))
        self._transition = {
            str(k): {str(w): float(p) for w, p in v.items()}
            for k, v in data.get("transition", {}).items()
        }
        self._word_weight = {str(k): float(v) for k, v in data.get("word_weight", {}).items()}
        self._starts = {str(k): float(v) for k, v in data.get("starts", {}).items()}
        self._trained = int(data.get("trained", 0))
