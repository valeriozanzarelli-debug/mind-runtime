"""Balbettio — vocalizzazione emergente da sillabe udite, non da liste fisse."""

from __future__ import annotations

import random

from organism.teaching.phonemes import PhonemeLearner


class BabbleMotor:
    def __init__(self, brain, seed: int = 0) -> None:
        self.brain = brain
        self.rng = random.Random(seed)
        self.phonemes = PhonemeLearner()
        self.phoneme_neurons = brain.get_neurons("motor", "speech_phoneme_generator")

    def hear(self, text: str, *, boost: float = 1.0) -> list[str]:
        heard = self.phonemes.hear(text, boost=boost)
        for i, syl in enumerate(heard[:8]):
            if i < len(self.phoneme_neurons):
                self.phoneme_neurons[i].activation = min(1.0, self.phoneme_neurons[i].activation + 0.15 * boost)
        return heard

    def utter(self, *, impulse: str = "vocalize", curiosity: float = 0.5) -> str:
        n = 1 + int(curiosity * 3)
        parts = self.phonemes.sample(n, self.rng, curiosity=curiosity)
        if not parts:
            return ""
        text = "-".join(parts)
        if impulse == "ask":
            text = text + "?"
        return text
