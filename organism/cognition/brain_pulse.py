"""Cervello sempre acceso — impulsi di fondo senza loop di parlato."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


class BrainPulse:
    """Thread daemon: propagate, affetti, persistenza leggera."""

    def __init__(
        self,
        tick_fn: Callable[[], dict[str, Any]],
        *,
        interval_s: float = 1.2,
        persist_every: int = 25,
    ) -> None:
        self._tick_fn = tick_fn
        self._interval = interval_s
        self._persist_every = persist_every
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.pulses = 0
        self._last: dict[str, Any] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="organism-brain-pulse", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._last = self._tick_fn()
                self.pulses += 1
            except Exception:
                pass

    def stats(self) -> dict[str, Any]:
        return {"pulses": self.pulses, "interval_s": self._interval, "last": self._last}
