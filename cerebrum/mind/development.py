"""Stadi di sviluppo — l'intelligenza si coltiva, non si programma.

Come un bambino, CEREBRUM non nasce con tutte le capacita' accese: le sblocca
in stadi, in base all'eta' (tick vissuti), all'esperienza (episodi) e al
vocabolario appreso. Ogni stadio abilita comportamenti nuovi del motore vocale
e della cognizione.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DevelopmentStage:
    level: int
    name: str
    description: str


STAGES = [
    DevelopmentStage(0, "neonato", "riflessi e pianto; nessun controllo volontario"),
    DevelopmentStage(1, "lallazione", "vocalizzi spontanei, sillabe canoniche"),
    DevelopmentStage(2, "imitazione", "ripete suoni/parole sentiti dal caregiver"),
    DevelopmentStage(3, "prime parole", "usa parole conosciute in modo intenzionale"),
    DevelopmentStage(4, "combinazioni", "combina piu' parole, frasi semplici"),
]


class Development:
    def __init__(self):
        self.level = 0
        self.experience = 0  # episodi/interazioni accumulate

    def update(self, *, age_ticks: int, vocabulary: int, episodes: int,
               mean_activity: float) -> int:
        """Valuta se avanzare di stadio. Ritorna il livello corrente."""
        self.experience = episodes
        # soglie combinate: tempo vissuto + parole sentite + attivita' cerebrale
        thresholds = [
            (1, age_ticks > 300),
            (2, age_ticks > 1200 and vocabulary >= 3),
            (3, age_ticks > 3000 and vocabulary >= 12 and episodes >= 20),
            (4, age_ticks > 8000 and vocabulary >= 40 and episodes >= 60),
        ]
        for lvl, cond in thresholds:
            if cond and self.level < lvl:
                self.level = lvl
        return self.level

    @property
    def stage(self) -> DevelopmentStage:
        return STAGES[min(self.level, len(STAGES) - 1)]

    def as_dict(self) -> dict:
        s = self.stage
        return {
            "level": s.level,
            "name": s.name,
            "description": s.description,
            "experience": self.experience,
        }
