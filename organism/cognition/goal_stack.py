"""Goal stack persistente — obiettivi emergono da stati indesiderati, decay lento."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Goal:
    label: str
    urgency: float = 0.5
    created_t: float = field(default_factory=time.time)
    source: str = ""

    def decayed_urgency(self, *, half_life_s: float = 7200.0) -> float:
        age_h = (time.time() - self.created_t) / max(1.0, half_life_s)
        return self.urgency * (0.5 ** age_h)


class GoalStack:
    """Obiettivi attivi tra cicli — bias attenzione verso goal corrente."""

    def __init__(self, *, max_goals: int = 12, half_life_s: float = 7200.0) -> None:
        self.max_goals = max_goals
        self.half_life_s = half_life_s
        self._goals: list[Goal] = []

    def emerge_from_state(
        self,
        *,
        uncertainty: float = 0.0,
        failure: str = "",
        boredom: float = 0.0,
        unknown_word: str = "",
    ) -> Goal | None:
        """Genera goal da stati indesiderati osservati."""
        if unknown_word:
            g = Goal(label=f"capire:{unknown_word}", urgency=0.7, source="unknown")
            self._push(g)
            return g
        if failure:
            g = Goal(label=f"riprovare:{failure[:32]}", urgency=0.75, source="failure")
            self._push(g)
            return g
        if uncertainty > 0.65:
            g = Goal(label="ridurre incertezza", urgency=uncertainty, source="uncertainty")
            self._push(g)
            return g
        if boredom > 0.7:
            g = Goal(label="esplorare qualcosa di nuovo", urgency=boredom * 0.8, source="boredom")
            self._push(g)
            return g
        return None

    def _push(self, goal: Goal) -> None:
        for i, g in enumerate(self._goals):
            if g.label == goal.label:
                self._goals[i] = Goal(
                    label=goal.label,
                    urgency=min(1.0, g.urgency + 0.15),
                    created_t=g.created_t,
                    source=goal.source,
                )
                return
        self._goals.append(goal)
        self._goals.sort(key=lambda x: -x.decayed_urgency(half_life_s=self.half_life_s))
        self._goals = self._goals[: self.max_goals]

    def active_goal(self) -> Goal | None:
        if not self._goals:
            return None
        self._goals.sort(key=lambda x: -x.decayed_urgency(half_life_s=self.half_life_s))
        g = self._goals[0]
        if g.decayed_urgency(half_life_s=self.half_life_s) < 0.05:
            self._goals.pop(0)
            return self.active_goal()
        return g

    def attention_bias_words(self) -> list[str]:
        g = self.active_goal()
        if not g:
            return []
        parts = g.label.replace(":", " ").split()
        return [p for p in parts if len(p) > 2][:4]

    def note_success(self, label: str) -> None:
        self._goals = [g for g in self._goals if g.label != label and label not in g.label]

    def stats(self) -> dict[str, Any]:
        ag = self.active_goal()
        return {
            "count": len(self._goals),
            "active": ag.label if ag else "",
            "urgency": round(ag.decayed_urgency(half_life_s=self.half_life_s), 3) if ag else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "half_life_s": self.half_life_s,
            "goals": [
                {"label": g.label, "urgency": g.urgency, "created_t": g.created_t, "source": g.source}
                for g in self._goals
            ],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self.half_life_s = float(data.get("half_life_s", self.half_life_s))
        self._goals = [
            Goal(
                label=str(g.get("label", "")),
                urgency=float(g.get("urgency", 0.5)),
                created_t=float(g.get("created_t", time.time())),
                source=str(g.get("source", "")),
            )
            for g in data.get("goals", [])
            if g.get("label")
        ]
