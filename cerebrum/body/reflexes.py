"""Riflessi neonatali — i comportamenti automatici presenti alla nascita.

Un neonato umano nasce con riflessi primitivi (Moro/trasalimento, rooting,
suzione, orientamento, pianto). Qui sono archi stimolo->risposta rapidi che
scavalcano la cognizione lenta, esattamente come nel tronco encefalico.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Reflex:
    name: str
    description: str


@dataclass
class NeonatalReflexes:
    reflexes: List[Reflex] = field(default_factory=lambda: [
        Reflex("moro", "trasalimento a stimoli improvvisi/rumore forte"),
        Reflex("rooting", "orientamento verso il tocco sulla guancia"),
        Reflex("suzione", "suzione in risposta a stimolo orale"),
        Reflex("grasp", "chiusura della mano sul contatto palmare"),
        Reflex("orienting", "orientamento di sguardo verso movimento/luce"),
        Reflex("crying", "pianto quando il distress è alto"),
    ])

    def evaluate(self, stimuli: dict, homeostasis: dict) -> List[dict]:
        """Ritorna i riflessi scattati in questo istante."""
        fired = []
        intensity = float(stimuli.get("intensity", 0.0))
        novelty = float(stimuli.get("novelty", 0.0))
        motion = float(stimuli.get("motion", 0.0))
        brightness = float(stimuli.get("brightness", 0.0))
        distress = float(homeostasis.get("distress", 0.0))

        if intensity > 0.7 or novelty > 0.85:
            fired.append({"reflex": "moro", "strength": round(max(intensity, novelty), 2)})
        if motion > 0.4 or brightness > 0.6:
            fired.append({"reflex": "orienting", "strength": round(max(motion, brightness), 2)})
        if distress > 0.6:
            fired.append({"reflex": "crying", "strength": round(distress, 2)})
        return fired

    def as_list(self) -> List[dict]:
        return [{"name": r.name, "description": r.description} for r in self.reflexes]
