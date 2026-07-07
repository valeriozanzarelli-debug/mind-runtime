"""Motore di vocalizzazione — dalla lallazione alle prime parole.

Come un neonato: all'inizio produce sillabe casuali (canonical babbling)
modulate dall'affetto e dall'attivazione del campo neurale. Man mano che
sente token ricorrenti, inizia a ri-emetterli (imitazione), poi a combinarli.
Non è un LLM: è emergenza fonetica guidata dallo stato interno.
"""
from __future__ import annotations

import random

_CONSONANTS = list("bdgmnptkl")
_VOWELS = list("aeiou")


class SpeechMotor:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.utterances = 0

    def _syllable(self) -> str:
        return self.rng.choice(_CONSONANTS) + self.rng.choice(_VOWELS)

    def babble(self, arousal: float, syllables: int = 0) -> str:
        n = syllables or max(1, int(1 + arousal * 3))
        return "".join(self._syllable() for _ in range(n))

    def utter(self, *, emotion: str, drive: str, activity: float,
              known_tokens=None, urge: float = 0.5) -> str:
        """Produce una vocalizzazione. Miscela lallazione, imitazione ed
        esclamazioni affettive in base allo stato."""
        known_tokens = known_tokens or []
        self.utterances += 1
        arousal = min(max(activity * 4.0 + urge, 0.1), 1.0)

        # esclamazioni riflesse per stati forti
        if emotion == "angoscia":
            return (self.babble(arousal, 2) + "!") if self.rng.random() < 0.5 else "uè uè"
        if emotion == "gioia":
            base = self.babble(arousal, self.rng.randint(1, 2))
            return base + (self.rng.choice(["!", "~", "-", ""]))
        if emotion == "attaccamento" and known_tokens:
            return self.rng.choice(known_tokens)

        # imitazione crescente: se ha sentito parole, a volte le ripete
        if known_tokens and self.rng.random() < min(0.15 + 0.4 * urge, 0.6):
            tok = self.rng.choice(known_tokens)
            if self.rng.random() < 0.3:
                return tok + " " + self.babble(arousal, 1)
            return tok

        # lallazione di default
        return self.babble(arousal, self.rng.randint(1, 3))
