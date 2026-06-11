"""Tipi condivisi tra i moduli del baby agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Madrid")

_COLOR_WORDS = frozenset(
    {"rosso", "verde", "blu", "giallo", "nero", "bianco", "grigio", "rosa", "arancione", "marrone"}
)


def _tokens_from_heard(heard: str) -> list[str]:
    stop = frozenset({"che", "chi", "come", "cosa", "il", "la", "un", "una", "di", "è", "era", "sono", "sei"})
    return [w for w in re.findall(r"[a-zàèéìòù']+", heard.lower()) if len(w) > 2 and w not in stop]


@dataclass
class BabyMoment:
    impulse: str
    spoke: str
    stimulus_key: str
    curiosity: dict[str, Any]
    learned: bool
    brain: dict[str, Any]
    thought: dict[str, Any]
    wanted_to_speak: bool = False
    code: str = ""
    understood: bool = False
    from_thought: bool = False
    self_heard: bool = False
    speech_error: dict[str, Any] = field(default_factory=dict)
    consciousness: dict[str, Any] = field(default_factory=dict)
    self_state: dict[str, Any] = field(default_factory=dict)
    wave: dict[str, Any] = field(default_factory=dict)
    dream: dict[str, Any] = field(default_factory=dict)
    emotion: dict[str, Any] = field(default_factory=dict)
    social_tone: dict[str, Any] = field(default_factory=dict)
    presence: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    consciousness_stream: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "impulse": self.impulse,
            "spoke": self.spoke,
            "code": self.code,
            "understood": self.understood,
            "from_thought": self.from_thought,
            "self_heard": self.self_heard,
            "speech_error": self.speech_error,
            "consciousness": self.consciousness,
            "self": self.self_state,
            "wave": self.wave,
            "dream": self.dream,
            "emotion": self.emotion,
            "social_tone": self.social_tone,
            "presence": self.presence,
            "task": self.task,
            "consciousness_stream": self.consciousness_stream,
            "thought": self.thought,
            "stimulus_key": self.stimulus_key,
            "curiosity": self.curiosity,
            "learned": self.learned,
            "brain": self.brain,
            "wanted_to_speak": self.wanted_to_speak,
            "symbols": self.symbols,
        }


def normalize_dialogue_key(when: str) -> str:
    import hashlib

    return hashlib.sha256(when.strip().lower().encode()).hexdigest()[:12]
