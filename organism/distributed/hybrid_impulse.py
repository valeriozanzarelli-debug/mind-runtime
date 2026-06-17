"""Impulso ibrido — GPU remota con fallback CPU locale automatico."""

from __future__ import annotations

import os
import time
from typing import Any

from organism.brain.impulse_scaffold import ImpulseScaffold


class HybridImpulseScaffold:
    """Prova GPU remota; se offline degrada su campo locale senza crash."""

    def __init__(
        self,
        remote_url: str | None = None,
        *,
        local_device: str = "auto",
        health_ttl_s: float = 12.0,
    ) -> None:
        self._remote_url = (remote_url or os.environ.get("ORGANISM_GPU_REMOTE", "")).strip() or None
        self._local_device = os.environ.get("ORGANISM_IMPULSE_DEVICE", local_device)
        self._health_ttl = health_ttl_s
        self._remote = None
        self._local: ImpulseScaffold | None = None
        self._mode = "none"
        self._last_health_t = 0.0
        self._remote_ok = False
        self._failures = 0
        self._pulse_count = 0
        if self._remote_url:
            from organism.distributed.remote_impulse import RemoteImpulseScaffold

            self._remote = RemoteImpulseScaffold(self._remote_url)
            self._mode = "remote"

    @property
    def last_reading(self):
        return self._active().last_reading

    def _active(self) -> Any:
        if self._mode == "remote" and self._remote is not None:
            return self._remote
        if self._local is None:
            try:
                self._local = ImpulseScaffold(device=self._local_device)
            except Exception:
                self._local = ImpulseScaffold(device="cpu")
            self._mode = "local_fallback"
        return self._local

    def _ensure_local(self) -> ImpulseScaffold:
        if self._local is None:
            try:
                self._local = ImpulseScaffold(device=self._local_device)
            except Exception:
                self._local = ImpulseScaffold(device="cpu")
        return self._local

    def _check_remote_health(self) -> bool:
        if not self._remote:
            return False
        now = time.time()
        if now - self._last_health_t < self._health_ttl:
            return self._remote_ok
        self._last_health_t = now
        try:
            import urllib.request

            with urllib.request.urlopen(f"{self._remote_url}/health", timeout=2.5) as resp:
                self._remote_ok = resp.status == 200
        except Exception:
            self._remote_ok = False
        return self._remote_ok

    def perceive_visual(self, gray: list | None, *, gain: float = 0.9) -> None:
        self._active().perceive_visual(gray, gain=gain)
        if self._local and self._mode == "remote":
            self._local.perceive_visual(gray, gain=gain * 0.5)

    def perceive_audio(self, bands: list[float] | None) -> None:
        self._active().perceive_audio(bands)

    def perceive_text(self, text: str | None) -> None:
        self._active().perceive_text(text)

    def pulse(self, *, steps: int = 2):
        self._pulse_count += 1
        if self._remote and self._mode == "remote":
            if not self._check_remote_health():
                self._mode = "local_fallback"
            else:
                try:
                    return self._remote.pulse(steps=steps)
                except Exception:
                    self._failures += 1
                    self._remote_ok = False
                    self._mode = "local_fallback"
        local = self._ensure_local()
        self._mode = "local_fallback"
        return local.pulse(steps=steps)

    def themes_for_speech(self) -> list[str]:
        return list(self._active().themes_for_speech())

    def symbols_for_mind(self) -> list[str]:
        return list(self._active().symbols_for_mind())

    def workspace_overlay(self) -> dict[str, Any]:
        fn = getattr(self._active(), "workspace_overlay", None)
        return fn() if callable(fn) else {}

    def stats(self) -> dict[str, Any]:
        base = dict(self._active().stats())
        base["hybrid_mode"] = self._mode
        base["remote_url"] = self._remote_url
        base["remote_failures"] = self._failures
        base["pulses"] = self._pulse_count
        return base
