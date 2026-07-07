"""Omeostasi / interocezione — gli stati vitali di base di un neonato.

Fame, energia, stanchezza, temperatura, comfort. Derivano nel tempo e
generano 'distress' quando escono dalla zona di comfort: è la spinta
primaria che tiene il cervello attivo (come un neonato che si lamenta).
"""
from __future__ import annotations

from dataclasses import dataclass


def _clip(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class Homeostasis:
    satiety: float = 0.6     # 1 = sazio, 0 = affamato
    energy: float = 0.8      # riserve
    fatigue: float = 0.2     # pressione del sonno
    warmth: float = 0.6      # comfort termico (0.5 = ideale)
    comfort: float = 0.6     # comfort viscerale generale
    pain: float = 0.0        # dolore acuto

    def as_dict(self) -> dict:
        return {
            "satiety": round(self.satiety, 3),
            "energy": round(self.energy, 3),
            "fatigue": round(self.fatigue, 3),
            "warmth": round(self.warmth, 3),
            "comfort": round(self.comfort, 3),
            "pain": round(self.pain, 3),
            "distress": round(self.distress(), 3),
        }

    def distress(self) -> float:
        hunger = 1.0 - self.satiety
        tired = self.fatigue
        cold_heat = abs(self.warmth - 0.5) * 2.0
        return _clip(0.4 * hunger + 0.3 * tired + 0.15 * cold_heat +
                     0.1 * (1.0 - self.comfort) + 0.5 * self.pain)

    def tick(self, dt: float, asleep: bool) -> None:
        # metabolismo: consumo lento di energia e sazietà
        self.satiety = _clip(self.satiety - 0.0006 * dt)
        self.energy = _clip(self.energy - (0.0002 if asleep else 0.0005) * dt)
        if asleep:
            self.fatigue = _clip(self.fatigue - 0.004 * dt)
            self.energy = _clip(self.energy + 0.0008 * dt)
        else:
            self.fatigue = _clip(self.fatigue + 0.0004 * dt)
        # il dolore si attenua da solo
        self.pain = _clip(self.pain - 0.02 * dt)
        # la temperatura tende verso l'ideale
        self.warmth = _clip(self.warmth + 0.001 * (0.5 - self.warmth) * dt)

    # azioni di accudimento dal caregiver (via chat/API)
    def feed(self, amount: float = 0.4) -> None:
        self.satiety = _clip(self.satiety + amount)
        self.comfort = _clip(self.comfort + 0.1)

    def soothe(self, amount: float = 0.3) -> None:
        self.comfort = _clip(self.comfort + amount)
        self.pain = _clip(self.pain - amount)

    def warm(self, delta: float = 0.15) -> None:
        self.warmth = _clip(self.warmth + delta)
