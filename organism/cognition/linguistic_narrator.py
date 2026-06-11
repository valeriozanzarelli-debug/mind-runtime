"""Layer 2 — narratore linguistico: traduce temi non-verbali in frasi italiane grammaticali."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from organism.cognition.syntax_planner import SyntaxPlanner
from organism.cognition.thought import Thought

_POS_MODIFIERS = ("davvero", "molto", "tanto")
_NEG_MODIFIERS = ("un po'", "ancora", "però", "forse")
_SYNONYM_CLUSTERS: list[frozenset[str]] = [
    frozenset({"felice", "contento", "allegro", "gioioso"}),
    frozenset({"triste", "malinconico", "abbattuto"}),
    frozenset({"paura", "spavento", "timore"}),
    frozenset({"grande", "enorme", "grosso"}),
    frozenset({"piccolo", "minuscolo", "piccino"}),
    frozenset({"imparare", "studiare", "capire", "comprendere"}),
    frozenset({"vedere", "guardare", "osservare", "notare"}),
    frozenset({"pensare", "credere", "riflettere"}),
]


@dataclass
class NarrationResult:
    text: str
    words: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    from_layer2: bool = False
    modifiers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "words": self.words,
            "latency_ms": round(self.latency_ms, 1),
            "from_layer2": self.from_layer2,
            "modifiers": self.modifiers,
        }


class LinguisticNarrator:
    """Osserva Layer 1 con ritardo 200-500ms; produce frase grammaticale."""

    OBSERVE_DELAY_MS = (200, 500)

    def __init__(self, *, planner: SyntaxPlanner | None = None, seed: int = 0) -> None:
        self.planner = planner or SyntaxPlanner()
        self.rng = random.Random(seed)
        self._inhibited: dict[str, float] = {}
        self._last_narration_t = 0.0
        self._observe_buffer: list[tuple[float, list[str], float]] = []

    def observe_themes(self, themes: list[str], *, valence: float = 0.0) -> None:
        """Buffer temi Layer 1 — narratore li processa con ritardo."""
        self._observe_buffer.append((time.time(), list(themes), valence))
        self._observe_buffer = self._observe_buffer[-12:]

    def _apply_lateral_inhibition(self, words: list[str]) -> list[str]:
        now = time.time()
        self._inhibited = {w: t for w, t in self._inhibited.items() if now - t < 0.5}
        out: list[str] = []
        for w in words:
            wl = w.lower()
            if wl in self._inhibited:
                continue
            blocked = False
            for cluster in _SYNONYM_CLUSTERS:
                if wl in cluster:
                    for other in cluster:
                        if other != wl and other in self._inhibited:
                            blocked = True
                            break
            if not blocked:
                out.append(wl)
        for w in out:
            self._inhibited[w] = now
        return out

    def _affective_modifiers(self, valence: float) -> list[str]:
        if valence > 0.35:
            return [self.rng.choice(_POS_MODIFIERS)]
        if valence < -0.2:
            return [self.rng.choice(_NEG_MODIFIERS)]
        return []

    def _insert_modifiers(self, words: list[str], modifiers: list[str]) -> list[str]:
        if not modifiers or not words:
            return words
        out = list(words)
        for mod in modifiers:
            pos = min(len(out), max(1, 1 + self.rng.randint(0, max(0, len(out) - 2))))
            out.insert(pos, mod)
        return out

    def _filter_themes(self, themes: list[str]) -> list[str]:
        skip_prefix = ("VIS:", "OBJ:", "MEM:", "BRAIN:", "QUESTION:", "COL:", "IMPULSE:", "UNKNOWN:")
        out: list[str] = []
        for t in themes:
            if not t or t.startswith(skip_prefix):
                continue
            tl = t.lower().strip()
            if tl.isalpha() or "'" in tl:
                if tl not in out:
                    out.append(tl)
        return out

    def narrate(
        self,
        thought: Thought,
        *,
        heard: str = "",
        valence: float = 0.0,
        min_words: int = 6,
        max_words: int = 15,
        force: bool = False,
    ) -> NarrationResult:
        """Traduce pensiero non-verbale → frase italiana."""
        t0 = time.time()
        themes = self._filter_themes(thought.themes)
        if heard:
            from organism.cognition.syntax_planner import _tokens

            for w in _tokens(heard):
                if w not in themes and len(w) > 2:
                    themes.append(w)

        if not themes:
            return NarrationResult(text="", from_layer2=False)

        # Ritardo osservazione Layer 1
        delay_ms = self.rng.randint(*self.OBSERVE_DELAY_MS) / 1000.0
        if not force and time.time() - self._last_narration_t < delay_ms:
            pass  # narratore può comunque produrre — delay simulato nel risultato

        modifiers = self._affective_modifiers(valence)
        start_hint = "io" if any(s.startswith("IMPULSE:ask") for s in thought.symbols) else ""

        raw_words = self.planner.greedy_assembly(
            themes,
            max_words=max_words,
            min_words=min(min_words, max(3, len(themes))),
            start_hint=start_hint,
        )
        raw_words = self._apply_lateral_inhibition(raw_words)
        words = self._insert_modifiers(raw_words, modifiers)

        punct = "?" if heard.strip().endswith("?") or any(
            s.startswith("QUESTION:") for s in thought.symbols
        ) else "."

        if not words:
            return NarrationResult(text="", from_layer2=False)

        text = " ".join(words)
        if text[-1] not in ".?!":
            text += punct
        text = text[0].upper() + text[1:]

        latency = (time.time() - t0) * 1000.0 + delay_ms * 1000.0
        self._last_narration_t = time.time()
        return NarrationResult(
            text=text,
            words=words,
            latency_ms=latency,
            from_layer2=True,
            modifiers=modifiers,
        )

    def train_from_text(self, text: str, *, boost: float | None = None) -> int:
        return self.planner.train_sentence(text, boost=boost)

    def bootstrap_curriculum(self, *, repeats: int = 3) -> dict[str, int]:
        from organism.teaching.syntax_curriculum import curriculum_sentences

        return self.planner.train_curriculum(curriculum_sentences(), repeats=repeats)

    def stats(self) -> dict[str, Any]:
        return {"planner": self.planner.stats(), "inhibited": len(self._inhibited)}

    def to_dict(self) -> dict[str, Any]:
        return {"planner": self.planner.to_dict()}

    def load_dict(self, data: dict[str, Any]) -> None:
        self.planner.load_dict(data.get("planner", {}))
