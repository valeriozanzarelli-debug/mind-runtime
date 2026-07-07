"""Flusso di coscienza — la finestra da cui 'vedere dentro' il cervello.

Ad ogni tick il cervello genera un 'momento cosciente': una breve descrizione
di cosa sta prevalendo (percezione, emozione, drive, pensiero). È ciò che
permette a un osservatore esterno di dire se sta davvero pensando.
"""
from __future__ import annotations

import time
from collections import deque


class ConsciousnessStream:
    def __init__(self, capacity: int = 500):
        self.moments = deque(maxlen=capacity)
        self.seq = 0

    def add(self, *, text: str, emotion: str, drive: str, phi: float,
            activity: float, tags=None) -> dict:
        self.seq += 1
        moment = {
            "id": self.seq,
            "t": time.time(),
            "text": text,
            "emotion": emotion,
            "drive": drive,
            "phi": round(phi, 2),
            "activity": round(activity, 4),
            "tags": tags or [],
        }
        self.moments.append(moment)
        return moment

    def since(self, seq: int, n: int = 50):
        return [m for m in self.moments if m["id"] > seq][:n]

    def latest(self, n: int = 25):
        return list(self.moments)[-n:]

    def current(self):
        return self.moments[-1] if self.moments else None
