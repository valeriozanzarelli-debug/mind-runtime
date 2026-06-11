"""Presenza — il bambino può parlare quando vuole; niente stati che bloccano la voce."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Madrid")


@dataclass
class PresenceState:
    shame: float = 0.0
    comfort: float = 0.5
    openness: float = 0.5
    urge: float = 0.0
    speaks: bool = True
    mood: str = "awake"
    mood_label: str = "awake"
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shame": round(self.shame, 3),
            "comfort": round(self.comfort, 3),
            "openness": round(self.openness, 3),
            "urge": round(self.urge, 3),
            "speaks": self.speaks,
            "mood": self.mood,
            "mood_label": self.mood_label,
            "blockers": list(self.blockers),
        }


class HumanPresence:
    """Neonato — esprime ciò che pensa; la vergogna modula solo l'intensità, non tace."""

    def __init__(self, *, baby: bool = True) -> None:
        self.baby = baby
        self.state = PresenceState()
        self._last_spoke_t: float = 0.0
        self._caregiver_until: float = 0.0
        self._familiar_scenes: set[str] = set()

    def note_caregiver(self, *, duration: float = 12.0) -> None:
        self._caregiver_until = time.time() + duration

    def note_spoke(self) -> None:
        self._last_spoke_t = time.time()

    def evaluate(
        self,
        *,
        curiosity: float,
        novelty: float,
        boredom: float,
        stimulus_key: str,
        visual_energy: float = 0.0,
        has_learned_phrase: bool = False,
        impulse: str = "attend",
        wants_voice: bool = False,
    ) -> PresenceState:
        now = time.time()
        familiar = stimulus_key in self._familiar_scenes if stimulus_key else novelty < 0.4
        if stimulus_key:
            self._familiar_scenes.add(stimulus_key)

        comfort = 0.5 + (0.25 if familiar else 0.0)
        if self._caregiver_until > now:
            comfort = min(1.0, comfort + 0.25)
        comfort += max(0.0, 0.15 - visual_energy * 0.2)
        comfort = min(1.0, max(0.2, comfort))

        shame = max(0.0, 0.05 + novelty * 0.15 - (0.2 if familiar else 0.0))
        if has_learned_phrase:
            shame *= 0.5

        urge = (
            0.35 * curiosity
            + 0.25 * boredom
            + 0.2 * novelty
            + 0.15 * comfort
            + (0.2 if impulse == "vocalize" else 0.0)
            + (0.15 if impulse == "ask" else 0.0)
            + (0.25 if wants_voice else 0.0)
        )
        urge = min(1.0, max(0.05, urge))

        blockers: list[str] = []
        # Il bambino parla se ha impulso — nessun blocco hard
        speaks = (
            urge > 0.08
            or wants_voice
            or impulse in ("vocalize", "ask")
            or curiosity > 0.2
            or has_learned_phrase
            or self.baby
        )

        if shame > 0.7:
            blockers.append("shy")
        if urge < 0.1:
            blockers.append("quiet")

        mood = "speaking" if speaks and urge > 0.35 else "awake"
        label = "vuole parlare" if speaks and urge > 0.35 else "osserva"

        self.state = PresenceState(
            shame=shame,
            comfort=comfort,
            openness=comfort,
            urge=urge,
            speaks=speaks,
            mood=mood,
            mood_label=label,
            blockers=blockers,
        )
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.state.to_dict(),
            "familiar_scenes": len(self._familiar_scenes),
            "caregiver_near": self._caregiver_until > time.time(),
            "baby": self.baby,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self.baby = bool(data.get("baby", True))
        self._familiar_scenes = set(data.get("familiar_scenes", []))
        self._last_spoke_t = float(data.get("last_spoke_t", 0))
        self._caregiver_until = float(data.get("caregiver_until", 0))
        s = data.get("state", {})
        if s:
            self.state = PresenceState(
                shame=float(s.get("shame", 0)),
                comfort=float(s.get("comfort", 0.5)),
                openness=float(s.get("openness", 0.5)),
                urge=float(s.get("urge", 0)),
                speaks=bool(s.get("speaks", True)),
                mood=str(s.get("mood", "awake")),
                mood_label=str(s.get("mood_label", "")),
                blockers=list(s.get("blockers", [])),
            )
