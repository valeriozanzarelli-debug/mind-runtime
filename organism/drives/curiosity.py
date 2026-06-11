"""Istinto di curiosità — voler imparare ed esplorare."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from organism.cognition.amygdala import AmygdalaEngine


@dataclass
class CuriosityState:
    level: float = 0.5
    novelty: float = 0.0
    uncertainty: float = 0.0
    boredom: float = 0.0
    last_stimulus_key: str = ""
    seen_stimuli: set[str] = field(default_factory=set)
    last_activity_t: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": round(self.level, 3),
            "novelty": round(self.novelty, 3),
            "uncertainty": round(self.uncertainty, 3),
            "boredom": round(self.boredom, 3),
            "stimuli_seen": len(self.seen_stimuli),
        }


class CuriosityDrive:
    """
    Non sa nulla del mondo. Sa solo che:
    - cose nuove → curiosità alta
    - pattern incompleti → vuole completare
    - silenzio lungo → vuole esplorare / vocalizzare
    """

    def __init__(self) -> None:
        self.state = CuriosityState()

    def observe(self, stimulus_key: str, *, pattern_gap: bool = False, learned: bool = False) -> float:
        self.state.last_activity_t = time.time()
        is_new = stimulus_key not in self.state.seen_stimuli
        if is_new and stimulus_key:
            self.state.seen_stimuli.add(stimulus_key)
        self.state.novelty = 1.0 if is_new else 0.15
        self.state.uncertainty = 0.8 if pattern_gap else (0.2 if learned else 0.55)
        self.state.boredom = 0.0
        self.state.last_stimulus_key = stimulus_key
        self.state.level = min(
            1.0,
            0.35 * self.state.novelty + 0.4 * self.state.uncertainty + 0.25,
        )
        return self.state.level

    def tick_idle(self, idle_seconds: float) -> float:
        self.state.boredom = min(1.0, idle_seconds / 12.0)
        self.state.novelty = 0.1
        self.state.uncertainty = 0.4 + self.state.boredom * 0.4
        self.state.level = min(1.0, 0.5 + self.state.boredom * 0.45)
        return self.state.level

    def choose_impulse(self, *, amygdala: AmygdalaEngine | None = None) -> str:
        """Tipo impulso da curiosità — opzionalmente modulato dall'amigdala."""
        if self.state.boredom > 0.65:
            base = "vocalize"
        elif self.state.novelty > 0.7:
            base = "investigate"
        elif self.state.uncertainty > 0.6:
            base = "ask"
        else:
            base = "attend"
        if amygdala is not None and hasattr(amygdala, "modulate_impulse"):
            return amygdala.modulate_impulse(base, self.state)
        return base


def stimulus_key_from_sensory(
    *,
    vision_hash: str = "",
    audio_hash: str = "",
    text: str = "",
    include_text: bool = True,
) -> str:
    t = text.strip().lower() if include_text else ""
    raw = f"{vision_hash}|{audio_hash}|{t}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def stimulus_key_visual_context(*, vision_hash: str = "", audio_hash: str = "") -> str:
    """Chiave per insegnare: scena visiva/uditiva senza legare al testo detto."""
    return stimulus_key_from_sensory(vision_hash=vision_hash, audio_hash=audio_hash, include_text=False)
