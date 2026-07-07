"""Neurochimica — i neuromodulatori che regolano apprendimento, umore, arousal.

Sono gli stessi assi che un cervello umano usa dalla nascita. Ogni valore è
0..1 e rilassa verso un set-point, spostato dagli stimoli e dagli stati del corpo.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _clip(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class Neurochemistry:
    dopamine: float = 0.3       # ricompensa / novità -> apprendimento
    serotonin: float = 0.5      # tono dell'umore / sazietà
    noradrenaline: float = 0.3  # arousal / vigilanza
    gaba: float = 0.4           # inibizione / calma
    glutamate: float = 0.5      # eccitazione
    acetylcholine: float = 0.4  # attenzione / plasticità
    oxytocin: float = 0.3       # legame / presenza del caregiver
    cortisol: float = 0.2       # stress
    melatonin: float = 0.2      # sonno

    _setpoints: dict = field(default_factory=lambda: {
        "dopamine": 0.3, "serotonin": 0.5, "noradrenaline": 0.3,
        "gaba": 0.4, "glutamate": 0.5, "acetylcholine": 0.4,
        "oxytocin": 0.3, "cortisol": 0.2, "melatonin": 0.2,
    })

    def as_dict(self) -> dict:
        return {
            "dopamine": round(self.dopamine, 3),
            "serotonin": round(self.serotonin, 3),
            "noradrenaline": round(self.noradrenaline, 3),
            "gaba": round(self.gaba, 3),
            "glutamate": round(self.glutamate, 3),
            "acetylcholine": round(self.acetylcholine, 3),
            "oxytocin": round(self.oxytocin, 3),
            "cortisol": round(self.cortisol, 3),
            "melatonin": round(self.melatonin, 3),
        }

    def release(self, name: str, amount: float) -> None:
        setattr(self, name, _clip(getattr(self, name) + amount))

    def update(self, stimuli: dict, homeostasis: dict) -> None:
        # novità -> dopamina + acetilcolina (attenzione)
        novelty = float(stimuli.get("novelty", 0.0))
        self.dopamine = _clip(self.dopamine + 0.25 * novelty)
        self.acetylcholine = _clip(self.acetylcholine + 0.2 * novelty)

        # presenza del caregiver -> ossitocina + serotonina, meno cortisolo
        presence = float(stimuli.get("presence", 0.0))
        self.oxytocin = _clip(self.oxytocin + 0.15 * presence)
        self.serotonin = _clip(self.serotonin + 0.05 * presence)
        self.cortisol = _clip(self.cortisol - 0.05 * presence)

        # disagio corporeo (fame, dolore, stanchezza) -> cortisolo + noradrenalina.
        # Solo il disagio oltre una soglia di comfort spinge lo stress, così un
        # neonato accudito si calma invece di restare sempre in allarme.
        distress = float(homeostasis.get("distress", 0.0))
        excess = max(distress - 0.35, 0.0)
        self.cortisol = _clip(self.cortisol + 0.15 * excess)
        self.noradrenaline = _clip(self.noradrenaline + 0.1 * excess)
        self.serotonin = _clip(self.serotonin - 0.04 * excess)

        # stimolazione sensoriale intensa -> glutammato/arousal (transitorio)
        intensity = float(stimuli.get("intensity", 0.0))
        self.glutamate = _clip(self.glutamate + 0.12 * intensity)
        self.noradrenaline = _clip(self.noradrenaline + 0.08 * intensity)

        # pressione del sonno -> melatonina
        fatigue = float(homeostasis.get("fatigue", 0.0))
        self.melatonin = _clip(self.melatonin + 0.1 * (fatigue - 0.5))

        # rilassamento omeostatico verso i set-point (più deciso, così gli
        # stati transitori decadono e il cervello torna a una base calma)
        for k, sp in self._setpoints.items():
            v = getattr(self, k)
            setattr(self, k, _clip(v + 0.12 * (sp - v)))

    def emotion(self) -> str:
        """Etichetta affettiva emergente dallo stato chimico."""
        if self.cortisol > 0.6 and self.noradrenaline > 0.5:
            return "angoscia"
        if self.dopamine > 0.6 and self.serotonin > 0.5:
            return "gioia"
        if self.oxytocin > 0.6:
            return "attaccamento"
        if self.melatonin > 0.6:
            return "sonnolenza"
        if self.dopamine > 0.55:
            return "curiosità"
        if self.serotonin > 0.6:
            return "serenità"
        if self.cortisol > 0.5:
            return "disagio"
        return "quiete"
