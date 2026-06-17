"""Impalcatura fissa — inietta sensoriale, legge coscienza, non decide."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from organism.brain.impulse_consciousness import ConsciousnessReading, ImpulseConsciousness
from organism.brain.impulse_field import ImpulseField, create_impulse_field
from organism.brain.impulse_memory import ImpulseMemory


def _default_size() -> tuple[int, int, int]:
    w = int(os.environ.get("ORGANISM_IMPULSE_W", "512"))
    h = int(os.environ.get("ORGANISM_IMPULSE_H", "384"))
    d = int(os.environ.get("ORGANISM_IMPULSE_D", "0"))
    return w, h, d


def _text_energies(text: str, dims: int = 64) -> list[float]:
    if not text:
        return []
    out = [0.0] * dims
    for token in text.lower().split():
        h = hashlib.sha256(token.encode()).digest()
        for i in range(min(dims, len(h))):
            out[i % dims] += h[i] / 255.0
    mx = max(out) if out else 1.0
    return [v / mx for v in out]


class ImpulseScaffold:
    """Codice fisso: gestisce il mare, la coscienza osserva separatamente."""

    def __init__(self, *, device: str = "auto", width: int | None = None, height: int | None = None, depth: int | None = None) -> None:
        w, h, d = _default_size()
        if width is not None:
            w = width
        if height is not None:
            h = height
        if depth is not None:
            d = depth
        self.field = create_impulse_field(w, h, device=device, depth=d)
        self.consciousness = ImpulseConsciousness()
        self.memory = ImpulseMemory()
        self._last_reading: ConsciousnessReading | None = None
        self._pulse_count = 0

    @property
    def last_reading(self) -> ConsciousnessReading | None:
        return self._last_reading

    def perceive_visual(self, gray: list | None, *, gain: float = 0.9) -> None:
        if gray:
            self.field.inject_pixels(gray, gain=gain)

    def perceive_audio(self, bands: list[float] | None) -> None:
        if bands:
            self.field.inject_audio_band(bands)

    def perceive_text(self, text: str | None) -> None:
        if text:
            self.field.inject_text_energy(_text_energies(text))

    def pulse(self, *, steps: int = 2) -> ConsciousnessReading:
        """Tick fisico + lettura coscienza."""
        self.field.step(steps=steps)
        self._pulse_count += 1
        sig = self.field.signature()
        recalled = self.memory.recall(sig, k=3)
        for ep in recalled:
            self.field.inject_memory_echo(ep.signature, gain=0.12 * ep.strength)
        reading = self.consciousness.observe(
            self.field,
            recalled_memories=[e.to_dict() for e in recalled],
            pressure=min(1.0, self.field.flux_magnitude() * 0.8),
        )
        self._last_reading = reading
        if reading.conscious and reading.novelty > 0.35 and self._pulse_count % 8 == 0:
            label = reading.thoughts[0] if reading.thoughts else reading.focus_region
            self.memory.store(sig, regions=self.field.regional_energy(), label=label[:60], tick=self.field.tick)
        return reading

    def themes_for_speech(self) -> list[str]:
        """Impalcatura estrae temi — motor parla, coscienza non parla direttamente."""
        r = self._last_reading
        if not r:
            return []
        themes: list[str] = []
        for item in r.thoughts + r.sensations:
            clean = item.split(":", 1)[-1] if ":" in item else item
            if clean and clean not in themes:
                themes.append(clean[:32])
        for mem in r.memories_recalled:
            if mem not in themes:
                themes.append(mem[:32])
        return themes[:8]

    def symbols_for_mind(self) -> list[str]:
        r = self._last_reading
        return list(r.broadcast[:10]) if r else []

    def workspace_overlay(self) -> dict[str, Any]:
        """Compatibilità con GlobalWorkspace / UI esistente."""
        r = self._last_reading
        if not r:
            return {}
        return {
            "conscious": r.conscious,
            "ignition": r.ignition,
            "broadcast": r.broadcast,
            "focus": r.focus_region,
            "mode": r.mode,
            "self_signal": r.self_signal,
            "impulse": True,
        }

    def stats(self) -> dict[str, Any]:
        base = self.field.stats()
        base["pulses"] = self._pulse_count
        base["memory"] = self.memory.stats()
        if self._last_reading:
            base["consciousness"] = self._last_reading.to_dict()
        return base

    def energy_bytes(self) -> bytes:
        return self.field.to_energy_bytes()

    def phase_bytes(self) -> bytes | None:
        fn = getattr(self.field, "to_phase_bytes", None)
        return fn() if callable(fn) else None

    def to_dict(self) -> dict[str, Any]:
        return {"memory": self.memory.to_dict(), "stats": self.stats()}

    def load_dict(self, data: dict[str, Any]) -> None:
        if "memory" in data:
            self.memory.load_dict(data["memory"])
