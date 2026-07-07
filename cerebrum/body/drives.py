"""Drives — le spinte motivazionali innate.

Curiosità (novità/incertezza), attaccamento (presenza del caregiver),
esplorazione (noia). Convertono lo stato interno in un 'bisogno di agire'
che modula il pensiero e la vocalizzazione.
"""
from __future__ import annotations

from dataclasses import dataclass


def _clip(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class Drives:
    curiosity: float = 0.5
    attachment: float = 0.4
    exploration: float = 0.5
    boredom: float = 0.2

    def as_dict(self) -> dict:
        return {
            "curiosity": round(self.curiosity, 3),
            "attachment": round(self.attachment, 3),
            "exploration": round(self.exploration, 3),
            "boredom": round(self.boredom, 3),
        }

    def update(self, stimuli: dict, chem: dict) -> None:
        novelty = float(stimuli.get("novelty", 0.0))
        presence = float(stimuli.get("presence", 0.0))
        intensity = float(stimuli.get("intensity", 0.0))

        # la novità sazia la curiosità sul momento ma il calo di stimoli la riaccende
        self.curiosity = _clip(self.curiosity + 0.2 * novelty - 0.02)
        self.boredom = _clip(self.boredom + (0.03 if intensity < 0.05 else -0.1))
        self.curiosity = _clip(self.curiosity + 0.3 * self.boredom)
        self.attachment = _clip(self.attachment + 0.1 * presence - 0.01)
        self.exploration = _clip(0.5 * self.curiosity + 0.5 * self.boredom)

    def dominant(self) -> str:
        vals = {
            "curiosità": self.curiosity,
            "attaccamento": self.attachment,
            "esplorazione": self.exploration,
        }
        return max(vals, key=vals.get)

    def urge(self) -> float:
        """Quanto forte è la spinta ad agire/vocalizzare adesso."""
        return _clip(max(self.curiosity, self.exploration, self.attachment))
