"""Emozioni interne — emergono da emotion_modulator + tono sociale."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from organism.brain.topology import NeuralTopology
from organism.sensory.social_tone import SocialTone


@dataclass
class AffectState:
    joy: float = 0.2
    fear: float = 0.1
    sadness: float = 0.05
    anger: float = 0.0
    trust: float = 0.4
    shame: float = 0.1
    curiosity: float = 0.5
    dominant: str = "curious"
    label: str = "curioso"

    def to_dict(self) -> dict[str, Any]:
        return {
            "joy": round(self.joy, 3),
            "fear": round(self.fear, 3),
            "sadness": round(self.sadness, 3),
            "anger": round(self.anger, 3),
            "trust": round(self.trust, 3),
            "shame": round(self.shame, 3),
            "curiosity": round(self.curiosity, 3),
            "dominant": self.dominant,
            "label": self.label,
        }


class AffectiveEngine:
    """Sistema limbico computazionale — legge neuroni emotion_modulator."""

    DECAY = 0.04

    def __init__(self) -> None:
        self.state = AffectState()
        self._pulse_count = 0

    def read_from_brain(self, brain: NeuralTopology) -> dict[str, float]:
        emos = brain.get_neurons("associative", "emotion_modulator")
        if not emos:
            return {}
        act = sum(n.activation for n in emos) / len(emos)
        return {"modulator_mean": act}

    def inject_into_brain(self, brain: NeuralTopology) -> None:
        """Feedback affettivo → neuroni modulatori."""
        emos = brain.get_neurons("associative", "emotion_modulator")
        if not emos:
            return
        s = self.state
        drives = [
            (s.joy, 0.12),
            (s.fear, 0.1),
            (s.anger, 0.14),
            (s.trust, 0.08),
            (s.shame, 0.11),
            (s.curiosity, 0.09),
        ]
        for i, n in enumerate(emos):
            boost = drives[i % len(drives)][0] * drives[i % len(drives)][1]
            n.activation = min(1.0, max(0.02, n.activation * 0.92 + boost))

    def update_from_social(
        self,
        tone: SocialTone,
        *,
        correction_applied: bool = False,
        curiosity_level: float = 0.5,
    ) -> AffectState:
        s = self.state
        s.curiosity = 0.7 * s.curiosity + 0.3 * curiosity_level

        if tone.is_angry:
            s.fear = min(1.0, s.fear + 0.25)
            s.anger = min(1.0, s.anger + 0.15)
            s.trust = max(0.0, s.trust - 0.2)
            s.shame = min(1.0, s.shame + 0.18)
            s.joy = max(0.0, s.joy - 0.15)
        elif tone.is_correction:
            s.shame = min(1.0, s.shame + 0.22)
            s.sadness = min(1.0, s.sadness + 0.1)
            if correction_applied:
                s.trust = min(1.0, s.trust + 0.08)
        elif tone.is_praise or tone.is_warm:
            s.joy = min(1.0, s.joy + 0.28)
            s.trust = min(1.0, s.trust + 0.2)
            s.shame = max(0.0, s.shame - 0.15)
            s.fear = max(0.0, s.fear - 0.1)

        s.joy = max(0.0, min(1.0, s.joy + tone.valence * 0.08))
        s.fear = max(0.0, min(1.0, s.fear + max(0, -tone.valence) * 0.06 + tone.arousal * 0.05))

        self._set_dominant()
        return s

    def note_visual_emotion(self, emotion: str) -> None:
        """Emozione vista su un volto — apprendimento visivo-affettivo graduale."""
        emo = emotion.strip().lower()
        s = self.state
        if emo in ("felice", "gioia"):
            s.joy = min(1.0, s.joy + 0.12)
            s.sadness = max(0.0, s.sadness - 0.05)
        elif emo in ("triste", "tristezza"):
            s.sadness = min(1.0, s.sadness + 0.14)
            s.joy = max(0.0, s.joy - 0.06)
        elif emo in ("arrabbiato", "rabbia", "irritato"):
            s.anger = min(1.0, s.anger + 0.14)
        elif emo in ("spaventato", "paura"):
            s.fear = min(1.0, s.fear + 0.14)
        elif emo in ("sorpreso",):
            s.curiosity = min(1.0, s.curiosity + 0.1)
        elif emo in ("calmo", "neutro"):
            s.trust = min(1.0, s.trust + 0.06)
            s.fear = max(0.0, s.fear - 0.04)
        self._set_dominant()

    def pulse_tick(self, brain: NeuralTopology, *, idle: bool = False) -> None:
        """Decadimento lento + mantenimento — cervello sempre acceso."""
        self._pulse_count += 1
        s = self.state
        d = self.DECAY * (0.5 if idle else 1.0)
        for attr in ("joy", "fear", "sadness", "anger", "shame"):
            val = getattr(s, attr)
            setattr(s, attr, max(0.02, val - d * val))
        s.trust = max(0.1, min(1.0, s.trust - d * 0.02))
        s.curiosity = max(0.15, min(1.0, s.curiosity - d * 0.03))
        if idle:
            s.curiosity = min(1.0, s.curiosity + 0.01)
        self.inject_into_brain(brain)
        brain.propagate(steps=1)
        self._set_dominant()

    def _set_dominant(self) -> None:
        s = self.state
        dims = {
            "joy": s.joy,
            "fear": s.fear,
            "sadness": s.sadness,
            "anger": s.anger,
            "trust": s.trust,
            "shame": s.shame,
            "curious": s.curiosity,
        }
        dom = max(dims, key=dims.get)
        s.dominant = dom
        labels = {
            "joy": "felice",
            "fear": "spaventato",
            "sadness": "triste",
            "anger": "irritato",
            "trust": "fiducioso",
            "shame": "imbarazzato",
            "curious": "curioso",
        }
        s.label = labels.get(dom, dom)

    def load_dict(self, data: dict[str, Any]) -> None:
        s = data.get("state", data)
        self.state = AffectState(
            joy=float(s.get("joy", 0.2)),
            fear=float(s.get("fear", 0.1)),
            sadness=float(s.get("sadness", 0.05)),
            anger=float(s.get("anger", 0)),
            trust=float(s.get("trust", 0.4)),
            shame=float(s.get("shame", 0.1)),
            curiosity=float(s.get("curiosity", 0.5)),
            dominant=str(s.get("dominant", "curious")),
            label=str(s.get("label", "curioso")),
        )
        self._pulse_count = int(data.get("pulse_count", 0))

    def stats(self) -> dict[str, Any]:
        return {"state": self.state.to_dict(), "pulse_count": self._pulse_count}
