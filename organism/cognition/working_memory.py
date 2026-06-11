"""Memoria di lavoro — grafo di temi paralleli (non stack FIFO)."""

from __future__ import annotations

from typing import Any


class WorkingMemory:
    """Grafo di slot attivi — più stream pensiero simultanei."""

    def __init__(self, *, capacity: int = 7) -> None:
        self.capacity = max(3, capacity)
        self._nodes: dict[str, float] = {}
        self._edges: dict[str, dict[str, float]] = {}
        self._slots: list[list[str]] = []  # legacy compat

    def activate(self, themes: list[str], *, weight: float = 1.0, link_to: list[str] | None = None) -> None:
        now = weight
        for t in themes:
            tl = self._norm(t)
            if not tl:
                continue
            self._nodes[tl] = min(3.0, self._nodes.get(tl, 0.0) + now)
        anchors = [self._norm(t) for t in (link_to or themes[:2]) if self._norm(t)]
        for i, a in enumerate(anchors):
            for b in anchors[i + 1 :]:
                self._link(a, b, now * 0.5)
            for t in themes:
                tb = self._norm(t)
                if tb and tb != a:
                    self._link(a, tb, now * 0.35)
        self._decay_nodes()
        self._trim()

    def push(self, themes: list[str], *, heard: str = "") -> None:
        """Compat API — aggiorna grafo + slot legacy."""
        slot: list[str] = []
        for t in themes:
            tl = self._norm(t)
            if tl:
                slot.append(tl)
        for w in heard.lower().split():
            if len(w) > 2 and w not in slot:
                slot.append(w)
        if slot:
            self._slots.append(slot[:8])
            self._slots = self._slots[-self.capacity :]
        self.activate(slot, weight=1.0, link_to=slot[:2])

    def _norm(self, t: str) -> str:
        tl = t.lower().strip()
        if not tl or tl.startswith(("vis:", "obj:", "col:", "mem:", "brain:")):
            return ""
        return tl

    def _link(self, a: str, b: str, w: float) -> None:
        ab = self._edges.setdefault(a, {})
        ab[b] = min(2.0, ab.get(b, 0.0) + w)
        ba = self._edges.setdefault(b, {})
        ba[a] = min(2.0, ba.get(a, 0.0) + w * 0.8)

    def _decay_nodes(self, factor: float = 0.92) -> None:
        for k in list(self._nodes.keys()):
            self._nodes[k] *= factor
            if self._nodes[k] < 0.05:
                del self._nodes[k]

    def _trim(self) -> None:
        if len(self._nodes) <= self.capacity * 3:
            return
        ranked = sorted(self._nodes.items(), key=lambda x: -x[1])
        keep = {k for k, _ in ranked[: self.capacity * 3]}
        self._nodes = {k: v for k, v in self._nodes.items() if k in keep}

    def context_words(self, *, limit: int = 24) -> list[str]:
        ranked = sorted(self._nodes.items(), key=lambda x: -x[1])
        out = [w for w, _ in ranked[:limit]]
        if len(out) < limit // 2:
            for slot in self._slots:
                for w in slot:
                    if w not in out:
                        out.append(w)
        return out[:limit]

    def parallel_streams(self, *, min_weight: float = 0.2) -> list[list[str]]:
        """Cluster temi connessi — pensieri paralleli."""
        visited: set[str] = set()
        streams: list[list[str]] = []
        nodes = [n for n, w in self._nodes.items() if w >= min_weight]
        for n in sorted(nodes, key=lambda x: -self._nodes.get(x, 0)):
            if n in visited:
                continue
            cluster = [n]
            visited.add(n)
            for nb, ew in sorted(self._edges.get(n, {}).items(), key=lambda x: -x[1]):
                if nb not in visited and ew >= min_weight * 0.5:
                    cluster.append(nb)
                    visited.add(nb)
            if cluster:
                streams.append(cluster[:6])
        return streams[:4]

    def recent_set(self) -> set[str]:
        return set(self._nodes.keys()) | {w for slot in self._slots for w in slot}

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._slots = []

    def stats(self) -> dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "edges": sum(len(v) for v in self._edges.values()) // 2,
            "slots": len(self._slots),
            "capacity": self.capacity,
            "words": self.context_words(limit=16),
            "streams": len(self.parallel_streams()),
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._slots = [list(s) for s in data.get("slots", [])][-self.capacity :]
        self._nodes = {str(k): float(v) for k, v in data.get("nodes", {}).items()}
        self._edges = {
            str(k): {str(w): float(ew) for w, ew in v.items()}
            for k, v in data.get("edges", {}).items()
        }
        if not self._nodes and self._slots:
            for slot in self._slots:
                self.activate(slot, weight=0.8)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": self._slots[-self.capacity :],
            "nodes": dict(self._nodes),
            "edges": {k: dict(v) for k, v in self._edges.items()},
        }
