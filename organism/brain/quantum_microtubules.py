"""Substrato microtubuli — metafora Orch-OR (Penrose/Hameroff), CONTESTATA.

Non simula fisica quantistica reale: modella coerenza/decoerenza come
variabile d'ordine che può modulare soglie di coscienza nel Global Workspace.
Riferimenti: Tegmark 2000 (decoerenza rapida), Frontiers 2024 (test sperimentali aperti).

Abilitare con env ORGANISM_QUANTUM=1.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any


def quantum_enabled() -> bool:
    return os.environ.get("ORGANISM_QUANTUM", "1").strip().lower() not in ("0", "false", "off")


@dataclass
class MicrotubuleState:
    coherence: float = 0.0
    decoherence_rate: float = 0.85
    phase: float = 0.0
    collapse_count: int = 0
    last_collapse_t: float = 0.0
    last_moment: str = ""
    contested: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "coherence": round(self.coherence, 4),
            "decoherence_rate": round(self.decoherence_rate, 4),
            "phase": round(self.phase, 4),
            "collapse_count": self.collapse_count,
            "last_moment": self.last_moment[:80],
            "contested_substrate": self.contested,
        }


@dataclass
class QuantumMicrotubuleLayer:
    """Coerenza mesoscopica → momenti di riduzione orchestrata (computazionale)."""

    state: MicrotubuleState = field(default_factory=MicrotubuleState)
    temperature_c: float = 37.0
    _enabled: bool = field(default_factory=quantum_enabled)

    def tick(
        self,
        *,
        neural_activity: float = 0.0,
        workspace_ignition: float = 0.0,
        gamma_hz: float = 40.0,
        thought_seed: str = "",
    ) -> MicrotubuleState:
        if not self._enabled:
            self.state.coherence = 0.0
            return self.state

        s = self.state
        # Decoerenza termica stile Tegmark — scalata, non fisica
        thermal = min(0.98, 0.75 + (self.temperature_c - 20) * 0.008)
        s.decoherence_rate = thermal

        drive = 0.25 * neural_activity + 0.45 * workspace_ignition
        s.phase += 2 * math.pi * gamma_hz * 0.001
        s.coherence = _clamp(s.coherence * (1 - s.decoherence_rate * 0.04) + drive * 0.12)

        # Collasso quando coerenza supera soglia — «momento conscio» discreto
        if s.coherence > 0.62 and (time.time() - s.last_collapse_t) > 0.08:
            s.collapse_count += 1
            s.last_collapse_t = time.time()
            s.last_moment = thought_seed[:80] or f"collapse#{s.collapse_count}"
            s.coherence *= 0.35

        return s

    def consciousness_threshold_delta(self) -> float:
        """Abbassa leggermente la soglia workspace se coerenza alta (ipotesi Orch-OR)."""
        if not self._enabled:
            return 0.0
        return -0.04 * self.state.coherence

    def collapse_boost(self) -> float:
        if not self._enabled or self.state.collapse_count == 0:
            return 0.0
        age = time.time() - self.state.last_collapse_t
        if age > 0.5:
            return 0.0
        return 0.15 * (1 - age / 0.5)

    def stats(self) -> dict[str, Any]:
        return {"enabled": self._enabled, "state": self.state.to_dict()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "temperature_c": self.temperature_c,
            "state": self.state.to_dict(),
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._enabled = bool(data.get("enabled", self._enabled))
        self.temperature_c = float(data.get("temperature_c", self.temperature_c))
        st = data.get("state", {})
        s = self.state
        s.coherence = float(st.get("coherence", s.coherence))
        s.decoherence_rate = float(st.get("decoherence_rate", s.decoherence_rate))
        s.phase = float(st.get("phase", s.phase))
        s.collapse_count = int(st.get("collapse_count", s.collapse_count))
        s.last_moment = str(st.get("last_moment", s.last_moment))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
