"""Lessico neurale — una parola è nota se ha un percorso sensoriale/motorio, non una tabella."""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from organism.brain.topology import NeuralTopology
    from organism.motor.emergent_speech import EmergentSpeechMotor

_TOKEN_RE = re.compile(r"[a-zàèéìòù']+")
_CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
EXPOSURE_SOFT_CAP = 80.0
EXPOSURE_HARD_CAP = 120.0

FILLER_WORDS = frozenset({
    "e", "o", "a", "il", "lo", "la", "i", "gli", "le", "un", "una", "uno",
    "che", "di", "da", "in", "con", "su", "per", "non", "mi", "ti", "si",
    "è", "sono", "ho", "hai", "ha", "c", "l", "d", "questo", "questa", "quello",
    "cosa", "come", "quando", "dove", "perché", "perche", "molto", "poco", "bene",
    "allora", "anche", "ancora", "solo", "già", "gia", "poi", "qui", "lì", "li",
})


def _tokens(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if len(w) >= 2]


class NeuralLexicon:
    """Parole legate a neuroni sensoriali — articolabilità = attivazione + esposizione Hebbiana."""

    def __init__(self) -> None:
        self._brain: NeuralTopology | None = None
        self._motor: EmergentSpeechMotor | None = None
        self._bindings: dict[str, int] = {}
        self._exposure: dict[str, float] = {}
        self._next_slot = 0

    def bind(self, brain: NeuralTopology, motor: EmergentSpeechMotor | None = None) -> None:
        self._brain = brain
        self._motor = motor
        for word, exp in list(self._exposure.items()):
            self._touch_word(word, boost=max(0.15, exp * 0.05))

    @property
    def count(self) -> int:
        return len(self._exposure)

    def known_words(self, limit: int = 24) -> list[str]:
        ranked = sorted(self._exposure.keys(), key=lambda w: -self._exposure[w])
        return ranked[:limit]

    def active_words(self, limit: int = 8, *, min_act: float = 0.1) -> list[str]:
        """Parole con neuroni sensoriali attualmente accesi — readout del momento."""
        if not self._brain:
            return self.known_words(limit)
        scored: list[tuple[float, str]] = []
        for w, nid in self._bindings.items():
            neuron = self._brain.neurons.get(nid)
            if not neuron:
                continue
            # Contesto corrente (neurone) domina; esposizione è bias secondario.
            act = neuron.activation + self._exposure.get(w, 0.0) * 0.02
            if act >= min_act:
                scored.append((act, w))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [w for _, w in scored[:limit]]

    def absorb(self, text: str, *, boost: float = 1.0) -> None:
        for w in _tokens(text):
            cur = self._exposure.get(w, 0.0)
            delta = boost * (1.0 / (1.0 + cur / 45.0))
            self._exposure[w] = min(EXPOSURE_HARD_CAP, cur + delta)
            self._touch_word(w, boost=boost)

    def prime_word(self, word: str, *, boost: float = 0.5) -> None:
        """Riattiva il percorso neurale di una parola già esposta."""
        w = word.lower().strip()
        if not w:
            return
        self._touch_word(w, boost=boost)

    def _touch_word(self, word: str, *, boost: float) -> None:
        if not self._brain:
            return
        encoders = self._brain.get_neurons("sensory", "text_semantic_encoder")
        if not encoders:
            return
        nid = self._bindings.get(word)
        if nid is None or nid not in self._brain.neurons:
            idx = self._next_slot % len(encoders)
            self._next_slot += 1
            nid = encoders[idx].id
            self._bindings[word] = nid
        neuron = self._brain.neurons[nid]
        neuron.activation = min(1.0, neuron.activation + 0.18 * boost)
        if self._motor:
            self._motor.hear(word, boost=boost * 0.35)

    def activation(self, word: str) -> float:
        w = word.lower().strip()
        exp = self._exposure.get(w, 0.0)
        if not self._brain or w not in self._bindings:
            return exp * 0.05
        neuron = self._brain.neurons.get(self._bindings[w])
        if not neuron:
            return exp * 0.05
        motor_boost = 0.0
        if self._motor:
            for n in self._motor.phoneme_neurons[:12]:
                if n.activation > 0.08:
                    motor_boost = max(motor_boost, n.activation * 0.25)
        return neuron.activation + min(0.4, exp * 0.12) + motor_boost

    def is_articulable(self, word: str, *, min_exposure: float = 0.7) -> bool:
        w = word.lower().strip()
        if len(w) < 2 or len(w) > 14:
            return False
        if sum(1 for c in w if c in _CONSONANTS) > len(w) * 0.7:
            return False
        exp = self._exposure.get(w, 0.0)
        if exp >= min_exposure:
            return True
        return self.activation(w) > 0.14

    def has_wired(self, *words: str, min_exposure: float = 0.5) -> bool:
        return all(self._exposure.get(w.lower(), 0.0) >= min_exposure or self.is_articulable(w) for w in words)

    def _score_word(self, word: str, *, avoid: set[str] | None = None) -> float:
        wl = word.lower().strip()
        exp = self._exposure.get(wl, 0.0)
        score = self.activation(wl) + min(exp, EXPOSURE_SOFT_CAP) * 0.12
        if exp > EXPOSURE_SOFT_CAP:
            score -= (exp - EXPOSURE_SOFT_CAP) * 0.025
        if avoid and wl in avoid:
            score *= 0.32
        return score

    def ranked(self, candidates: list[str], *, avoid: list[str] | None = None) -> list[str]:
        avoid_s = {w.lower() for w in (avoid or [])}
        scored = [(self._score_word(w, avoid=avoid_s), w) for w in candidates]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out: list[str] = []
        for _, w in scored:
            wl = w.lower()
            if self.is_articulable(wl) and wl not in out:
                out.append(wl)
        return out

    def sample(
        self,
        n: int,
        rng: random.Random,
        *,
        prefer: list[str] | None = None,
        avoid: list[str] | None = None,
    ) -> list[str]:
        pool = list(dict.fromkeys((prefer or []) + list(self._exposure.keys())))
        if not pool:
            return []
        avoid_s = {w.lower() for w in (avoid or [])}
        weights = [max(0.02, self._score_word(w, avoid=avoid_s)) for w in pool]
        return [rng.choices(pool, weights=weights, k=1)[0].lower() for _ in range(max(1, n))]

    def squash_overexposed(self, *, cap: float = EXPOSURE_SOFT_CAP) -> int:
        """Comprime parole iper-addestrate — una tantum dopo migrazione."""
        n = 0
        for w, exp in list(self._exposure.items()):
            if exp > cap:
                self._exposure[w] = cap + (exp - cap) ** 0.55
                n += 1
        return n

    def to_dict(self) -> dict[str, Any]:
        return {
            "bindings": dict(self._bindings),
            "exposure": dict(self._exposure),
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        if "exposure" in data:
            self._bindings = {str(k): int(v) for k, v in data.get("bindings", {}).items()}
            self._exposure = {str(k): float(v) for k, v in data.get("exposure", {}).items()}
            self._next_slot = len(self._bindings)
            return
        # Migrazione da WordLearner / word_weights legacy
        legacy = data.get("weights", {})
        self._exposure = {str(k): float(v) for k, v in legacy.items()}
        self._bindings = {}
        self._next_slot = 0
