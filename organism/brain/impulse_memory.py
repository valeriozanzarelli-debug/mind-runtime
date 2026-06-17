"""Memoria episodica del mare impulsi — impronte spazio-temporali."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImpulseEpisode:
    id: str
    signature: list[float]
    label: str
    regions: dict[str, float]
    tick: float = 0.0
    strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "tick": self.tick,
            "strength": round(self.strength, 4),
            "regions": {k: round(v, 4) for k, v in self.regions.items()},
        }


class ImpulseMemory:
    """Richiama episodi per similarità di impronta — non è la coscienza."""

    def __init__(self, *, capacity: int = 120) -> None:
        self.capacity = capacity
        self._episodes: list[ImpulseEpisode] = []
        self._counter = 0

    def store(
        self,
        signature: list[float],
        *,
        regions: dict[str, float],
        label: str = "",
        tick: float = 0.0,
    ) -> ImpulseEpisode:
        self._counter += 1
        ep = ImpulseEpisode(
            id=f"imp_{self._counter}",
            signature=signature[:128],
            label=label or f"episodio_{self._counter}",
            regions=dict(regions),
            tick=tick,
        )
        self._episodes.append(ep)
        if len(self._episodes) > self.capacity:
            self._episodes = self._episodes[-self.capacity :]
        return ep

    def recall(self, signature: list[float], *, k: int = 3, min_sim: float = 0.55) -> list[ImpulseEpisode]:
        if not signature or not self._episodes:
            return []
        scored: list[tuple[float, ImpulseEpisode]] = []
        for ep in self._episodes:
            sim = _cosine(signature, ep.signature)
            if sim >= min_sim:
                scored.append((sim * ep.strength, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def reinforce(self, episode_id: str, *, boost: float = 0.15) -> None:
        for ep in self._episodes:
            if ep.id == episode_id:
                ep.strength = min(2.0, ep.strength + boost)
                return

    def stats(self) -> dict[str, Any]:
        return {"episodes": len(self._episodes), "capacity": self.capacity}

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": self.stats(),
            "recent": [e.to_dict() for e in self._episodes[-12:]],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._episodes = []
        for raw in data.get("recent", []):
            self._episodes.append(
                ImpulseEpisode(
                    id=str(raw.get("id", "imp_0")),
                    signature=[],
                    label=str(raw.get("label", "")),
                    regions=dict(raw.get("regions", {})),
                    tick=float(raw.get("tick", 0)),
                    strength=float(raw.get("strength", 1.0)),
                )
            )


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = sum(a[i] * a[i] for i in range(n)) ** 0.5
    nb = sum(b[i] * b[i] for i in range(n)) ** 0.5
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)
