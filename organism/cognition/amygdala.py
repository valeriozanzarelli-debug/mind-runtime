"""Amigdala — salienza, paura, gioco, esplorazione; modula curiosità e parlato."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from organism.brain.topology import NeuralTopology
from organism.drives.curiosity import CuriosityState


@dataclass
class AmygdalaState:
    approach: float = 0.45
    avoid: float = 0.12
    play: float = 0.25
    explore: float = 0.5
    salience: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach": round(self.approach, 3),
            "avoid": round(self.avoid, 3),
            "play": round(self.play, 3),
            "explore": round(self.explore, 3),
            "salience": round(self.salience, 3),
        }


class AmygdalaEngine:
    """
    Proxy limbico su emotion_modulator — nessuna parola fissa.
    Integra curiosità (drive) + affetto (gioia/paura) in impulsi comportamentali.
    """

    def __init__(self) -> None:
        self.state = AmygdalaState()

    def pulse(
        self,
        brain: NeuralTopology,
        *,
        joy: float,
        fear: float,
        shame: float,
        anger: float,
        trust: float,
        curiosity: float,
        curiosity_drive: CuriosityState,
    ) -> AmygdalaState:
        emos = brain.get_neurons("associative", "emotion_modulator")
        mod_act = sum(n.activation for n in emos) / len(emos) if emos else 0.15

        novelty = curiosity_drive.novelty
        uncertainty = curiosity_drive.uncertainty
        boredom = curiosity_drive.boredom

        s = self.state
        s.avoid = min(1.0, fear * 0.55 + shame * 0.35 + anger * 0.25 + mod_act * 0.08)
        s.approach = min(1.0, joy * 0.45 + trust * 0.35 + curiosity * 0.35 + (1.0 - s.avoid) * 0.15)
        s.play = min(1.0, joy * 0.5 + novelty * 0.35 + (1.0 - boredom) * 0.2)
        s.explore = min(1.0, curiosity * 0.4 + uncertainty * 0.35 + novelty * 0.3 - s.avoid * 0.25)
        s.salience = min(1.0, uncertainty * 0.4 + novelty * 0.35 + mod_act * 0.25)
        return s

    def modulate_impulse(self, base: str, curiosity: CuriosityState) -> str:
        s = self.state
        if s.avoid > 0.62 and base in ("vocalize", "ask"):
            return "attend"
        if s.play > 0.55 and curiosity.boredom > 0.35:
            return "vocalize"
        if s.explore > 0.48 and curiosity.uncertainty > 0.45:
            return "ask"
        if s.approach > 0.6 and curiosity.novelty > 0.55:
            return "investigate"
        return base

    def speech_inhibition(self) -> float:
        return self.state.avoid * 0.12 + self.state.salience * 0.03

    def word_boost(self, word: str, *, activation: float) -> float:
        """Ranking lessicale — avvicinamento aumenta, paura inibisce parole deboli."""
        s = self.state
        boost = activation
        if s.approach > 0.45:
            boost += s.approach * 0.12
        if s.play > 0.4:
            boost += s.play * 0.08
        if s.avoid > 0.4 and activation < 0.35:
            boost *= max(0.25, 1.0 - s.avoid * 0.7)
        return boost

    def stats(self) -> dict[str, Any]:
        return {"state": self.state.to_dict()}

    def load_dict(self, data: dict[str, Any]) -> None:
        s = data.get("state", data)
        self.state = AmygdalaState(
            approach=float(s.get("approach", 0.45)),
            avoid=float(s.get("avoid", 0.12)),
            play=float(s.get("play", 0.25)),
            explore=float(s.get("explore", 0.5)),
            salience=float(s.get("salience", 0.3)),
        )
