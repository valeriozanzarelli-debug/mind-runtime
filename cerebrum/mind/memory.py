"""Memoria episodica — traccia degli eventi salienti con consolidamento.

Ogni episodio è salvato con salienza (novità * affetto). Nel sonno gli
episodi a bassa salienza vengono potati (pruning) e i restanti rinforzati.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Optional


class EpisodicMemory:
    def __init__(self, capacity: int = 5000, vault_path: Optional[str] = None):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.vault_path = vault_path
        self.consolidated = 0

    def store(self, kind: str, content: str, salience: float, meta: Optional[dict] = None) -> None:
        self.episodes.append({
            "t": time.time(),
            "kind": kind,
            "content": content,
            "salience": round(float(salience), 3),
            "meta": meta or {},
        })

    def recent(self, n: int = 20):
        return list(self.episodes)[-n:]

    def salient(self, n: int = 10):
        return sorted(self.episodes, key=lambda e: e["salience"], reverse=True)[:n]

    def recall(self, cue: str) -> Optional[dict]:
        cue = cue.lower()
        best, score = None, 0.0
        for e in self.episodes:
            overlap = len(set(cue.split()) & set(str(e["content"]).lower().split()))
            s = overlap + e["salience"]
            if overlap and s > score:
                best, score = e, s
        return best

    def consolidate(self, threshold: float = 0.2) -> int:
        """Pruning sinaptico del sonno: rimuove episodi poco salienti."""
        keep = [e for e in self.episodes if e["salience"] >= threshold]
        pruned = len(self.episodes) - len(keep)
        self.episodes = deque(keep, maxlen=self.capacity)
        self.consolidated += 1
        return pruned

    def save(self) -> None:
        if not self.vault_path:
            return
        try:
            os.makedirs(os.path.dirname(self.vault_path), exist_ok=True)
            with open(self.vault_path, "w", encoding="utf-8") as f:
                json.dump(list(self.episodes), f)
        except Exception:
            pass

    def load(self) -> None:
        if not self.vault_path or not os.path.exists(self.vault_path):
            return
        try:
            with open(self.vault_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.episodes = deque(data, maxlen=self.capacity)
        except Exception:
            pass

    def size(self) -> int:
        return len(self.episodes)
