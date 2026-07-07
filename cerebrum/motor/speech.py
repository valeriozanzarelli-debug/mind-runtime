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
              known_tokens=None, urge: float = 0.5, stage: int = 0) -> str:
        """Produce una vocalizzazione. Il repertorio dipende dallo STADIO DI
        SVILUPPO: 0 neonato (pianto/versi), 1 lallazione, 2 imitazione,
        3 prime parole intenzionali, 4 combinazioni di parole."""
        known_tokens = known_tokens or []
        self.utterances += 1
        arousal = min(max(activity * 4.0 + urge, 0.1), 1.0)

        # esclamazioni riflesse per stati forti (a ogni eta')
        if emotion == "angoscia":
            return (self.babble(arousal, 2) + "!") if self.rng.random() < 0.5 else "uè uè"

        # stadio 0: neonato — solo versi/pianto, nessun controllo
        if stage <= 0:
            return self.rng.choice(["uè", "aa", "eh", "ah", "ngh"])

        if emotion == "gioia":
            base = self.babble(arousal, self.rng.randint(1, 2))
            return base + (self.rng.choice(["!", "~", "-", ""]))

        # stadio >=2: imitazione crescente delle parole sentite
        imitation_p = {1: 0.0, 2: 0.35, 3: 0.55, 4: 0.7}.get(stage, 0.2)
        if emotion == "attaccamento" and known_tokens and stage >= 2:
            return self.rng.choice(known_tokens)
        if known_tokens and stage >= 2 and self.rng.random() < min(imitation_p + 0.2 * urge, 0.85):
            # stadio 4: combina due parole
            if stage >= 4 and len(known_tokens) >= 2 and self.rng.random() < 0.5:
                a, b = self.rng.sample(known_tokens, 2)
                return f"{a} {b}"
            tok = self.rng.choice(known_tokens)
            if stage == 2 and self.rng.random() < 0.4:
                return tok + " " + self.babble(arousal, 1)
            return tok

        # stadio 1 (e fallback): lallazione
        return self.babble(arousal, self.rng.randint(1, 3))
