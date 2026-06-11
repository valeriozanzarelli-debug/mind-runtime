"""Memoria episodica — breve (minuti) e lunga (giorni)."""

from __future__ import annotations

import time
from typing import Any


class EpisodicMemory:
    def __init__(
        self,
        *,
        short_slots: int = 24,
        long_capacity: int = 800,
        episodic_decay: float = 0.0005,
    ) -> None:
        self.short_slots = short_slots
        self.long_capacity = long_capacity
        self.episodic_decay = episodic_decay
        self._short: list[dict[str, Any]] = []
        self._long: list[dict[str, Any]] = []
        self._theme_salience: dict[str, float] = {}

    def record(
        self,
        *,
        heard: str = "",
        spoke: str = "",
        themes: list[str] | None = None,
        objects: list[str] | None = None,
        emotion: str = "",
    ) -> None:
        ep = {
            "t": time.time(),
            "heard": heard[:120],
            "spoke": spoke[:200],
            "themes": (themes or [])[:8],
            "objects": (objects or [])[:6],
            "emotion": emotion[:24],
        }
        self._short.append(ep)
        self._short = self._short[-self.short_slots :]
        for t in ep.get("themes", []):
            tl = str(t).lower().strip()
            if tl:
                self._theme_salience[tl] = min(3.0, self._theme_salience.get(tl, 0.0) + 0.15)
        if spoke and len(spoke) > 20:
            self._long.append(ep)
            self._long = self._long[-self.long_capacity :]
        self._decay_salience()

    def _decay_salience(self) -> None:
        for k in list(self._theme_salience.keys()):
            self._theme_salience[k] = max(0.0, self._theme_salience[k] - self.episodic_decay)
            if self._theme_salience[k] <= 0:
                del self._theme_salience[k]

    def recent_themes(self, *, limit: int = 12) -> list[str]:
        out: list[str] = []
        ranked = sorted(self._theme_salience.items(), key=lambda x: -x[1])
        for t, _ in ranked[:limit]:
            out.append(t)
        for ep in reversed(self._short):
            for t in ep.get("themes", []):
                if t not in out:
                    out.append(t)
            if len(out) >= limit:
                break
        return out[:limit]

    def consolidate_short_to_long(self, *, min_spoke_len: int = 12) -> int:
        """Consolida episodi brevi in memoria lunga — ogni ~50 coppie training."""
        moved = 0
        for ep in self._short:
            spoke = str(ep.get("spoke", ""))
            if len(spoke) >= min_spoke_len and ep not in self._long:
                self._long.append(ep)
                moved += 1
        self._long = self._long[-self.long_capacity :]
        return moved

    def recall_context(self, query: str = "", *, limit: int = 3) -> list[dict[str, Any]]:
        q = query.lower().strip()
        if not q:
            return self._short[-limit:]
        hits: list[dict[str, Any]] = []
        for ep in reversed(self._long + self._short):
            blob = f"{ep.get('heard','')} {ep.get('spoke','')} {' '.join(ep.get('themes',[]))}"
            if any(w in blob.lower() for w in q.split() if len(w) > 2):
                hits.append(ep)
            if len(hits) >= limit:
                break
        return hits

    def stats(self) -> dict[str, Any]:
        return {
            "short": len(self._short),
            "long": len(self._long),
            "short_slots": self.short_slots,
            "long_capacity": self.long_capacity,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "short": self._short,
            "long": self._long[-self.long_capacity :],
            "short_slots": self.short_slots,
            "long_capacity": self.long_capacity,
            "episodic_decay": self.episodic_decay,
            "theme_salience": dict(self._theme_salience),
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._short = list(data.get("short", []))[-self.short_slots :]
        self._long = list(data.get("long", []))[-self.long_capacity :]
        self.episodic_decay = float(data.get("episodic_decay", self.episodic_decay))
        self._theme_salience = {str(k): float(v) for k, v in data.get("theme_salience", {}).items()}
