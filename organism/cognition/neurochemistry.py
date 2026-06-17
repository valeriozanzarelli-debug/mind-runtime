"""Chimica sinaptica — pool di neurotrasmettitori che modulano affetto, plasticità, arousal.

Basato su modelli di neuromodulazione (dopamina/serotonina/NE/GABA/glutammato/ACh)
usati in simulazioni multiscala (TVB, mean-field). Non sostituisce spike singoli:
è un livello mesoscopico compatibile con emotion_modulator nel grafo DNA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from organism.brain.topology import NeuralTopology


@dataclass
class NeurochemicalState:
    """Concentrazioni normalizzate 0–1 nei compartimenti sinaptici."""

    dopamine: float = 0.48
    serotonin: float = 0.52
    norepinephrine: float = 0.35
    gaba: float = 0.42
    glutamate: float = 0.58
    acetylcholine: float = 0.4
    oxytocin: float = 0.32
    endorphin: float = 0.25
    adenosine: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


@dataclass
class NeurochemistryEngine:
    """Sistema monoaminergico + colinergico + peptidico semplificato."""

    state: NeurochemicalState = field(default_factory=NeurochemicalState)
    _pulse: int = 0
    _reuptake: float = 0.06

    def tick(
        self,
        *,
        joy: float = 0.2,
        fear: float = 0.1,
        anger: float = 0.0,
        trust: float = 0.4,
        curiosity: float = 0.5,
        stress: float = 0.0,
        social_warm: bool = False,
        idle_s: float = 0.0,
        learned: bool = False,
    ) -> NeurochemicalState:
        self._pulse += 1
        s = self.state
        r = self._reuptake

        # Ricompensa / apprendimento (mesolimbico)
        reward = 0.35 * joy + 0.25 * trust + (0.2 if learned else 0.0) + 0.15 * curiosity
        s.dopamine = _clamp(s.dopamine * (1 - r) + reward * 0.18)

        # Umore stabile / impulsività (raphe)
        mood = 0.5 * joy + 0.3 * trust - 0.25 * anger - 0.2 * fear
        s.serotonin = _clamp(s.serotonin * (1 - r) + (0.45 + mood * 0.35) * 0.12)

        # Arousal / allerta (locus coeruleus)
        threat = max(fear, anger, stress)
        s.norepinephrine = _clamp(
            s.norepinephrine * (1 - r) + (0.2 + threat * 0.55 + curiosity * 0.15) * 0.14
        )

        # Inibizione (interneuroni)
        s.gaba = _clamp(s.gaba * (1 - r * 0.8) + (0.35 + s.serotonin * 0.25 - threat * 0.2) * 0.1)

        # Eccitazione globale
        s.glutamate = _clamp(
            s.glutamate * (1 - r) + (0.4 + curiosity * 0.3 + s.norepinephrine * 0.2) * 0.12
        )

        # Attenzione / consolidamento (basale di Meynert)
        s.acetylcholine = _clamp(
            s.acetylcholine * (1 - r) + (0.3 + curiosity * 0.35 + (0.15 if learned else 0)) * 0.11
        )

        # Legame sociale (ipotalamo)
        if social_warm:
            s.oxytocin = _clamp(s.oxytocin + 0.08)
        s.oxytocin = _clamp(s.oxytocin * (1 - r * 0.5) + trust * 0.06)

        # Analgesia / piacere
        s.endorphin = _clamp(s.endorphin * (1 - r) + joy * 0.08)

        # Pressione omeostatica del sonno (adenosina)
        fatigue = min(1.0, idle_s / 120.0)
        s.adenosine = _clamp(s.adenosine * 0.97 + fatigue * 0.04)

        return s

    def plasticity_gain(self) -> float:
        """Moltiplicatore Hebbian — dopamina + ACh aumentano LTP."""
        s = self.state
        return 0.75 + 0.5 * s.dopamine + 0.35 * s.acetylcholine - 0.25 * s.gaba

    def arousal_bias(self) -> float:
        s = self.state
        return _clamp(0.25 * s.norepinephrine + 0.2 * s.glutamate - 0.15 * s.gaba + 0.1 * s.dopamine)

    def mood_valence(self) -> float:
        s = self.state
        return _clamp(0.4 * s.dopamine + 0.35 * s.serotonin + 0.15 * s.oxytocin - 0.3 * s.norepinephrine)

    def inject_into_brain(self, brain: NeuralTopology) -> None:
        """Neuromodulatori → emotion_modulator + pattern_matcher."""
        s = self.state
        emos = brain.get_neurons("associative", "emotion_modulator")
        for i, n in enumerate(emos):
            mod = [
                s.dopamine * 0.14,
                s.serotonin * 0.1,
                s.norepinephrine * 0.12,
                s.gaba * -0.06,
                s.glutamate * 0.08,
                s.acetylcholine * 0.09,
            ]
            n.activation = _clamp(n.activation * 0.94 + mod[i % len(mod)])

        matchers = brain.get_neurons("associative", "pattern_matcher")
        boost = self.arousal_bias() * 0.08
        for n in matchers[: max(1, int(len(matchers) * boost))]:
            n.activation = _clamp(n.activation + boost)

    def modulate_affect_dims(
        self,
        *,
        joy: float,
        fear: float,
        sadness: float,
        anger: float,
        trust: float,
        curiosity: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Feedback chimico → dimensioni affettive."""
        s = self.state
        v = self.mood_valence()
        return (
            _clamp(joy + v * 0.08 + s.dopamine * 0.05),
            _clamp(fear + s.norepinephrine * 0.1 - s.gaba * 0.05),
            _clamp(sadness + (1 - s.serotonin) * 0.06),
            _clamp(anger + s.norepinephrine * 0.08),
            _clamp(trust + s.oxytocin * 0.1 + s.serotonin * 0.04),
            _clamp(curiosity + s.dopamine * 0.08 + s.acetylcholine * 0.06),
        )

    def stats(self) -> dict[str, Any]:
        return {"state": self.state.to_dict(), "pulse": self._pulse}

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.to_dict(), "pulse": self._pulse}

    def load_dict(self, data: dict[str, Any]) -> None:
        st = data.get("state", data)
        self.state = NeurochemicalState(
            **{k: float(st.get(k, getattr(self.state, k))) for k in NeurochemicalState.__dataclass_fields__}
        )
        self._pulse = int(data.get("pulse", 0))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
