"""Global Workspace — coscienza come auto-percezione, non cancello al parlato."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from organism.brain.topology import NeuralTopology
from organism.cognition.thought import Thought

OutputMode = Literal["flow", "reflect", "speak"]


@dataclass
class WorkspaceState:
    conscious: bool
    ignition: float
    broadcast: list[str]
    focus: str
    mode: OutputMode = "flow"
    self_signal: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "conscious": self.conscious,
            "ignition": round(self.ignition, 3),
            "broadcast": self.broadcast[:10],
            "focus": self.focus,
            "mode": self.mode,
            "self_signal": round(self.self_signal, 3),
        }


class GlobalWorkspace:
    """Coscienza = percepire sé pensando — parla e riflette insieme (bambino)."""

    def __init__(self, *, threshold: float = 0.22) -> None:
        self.threshold = threshold
        self._ignitions = 0
        self._last_focus = ""
        self._self_continuity = 0.0

    def cycle(
        self,
        brain: NeuralTopology,
        thought: Thought,
        *,
        sensory_symbols: list[str] | None = None,
        novelty: float = 0.0,
        wave_phase: str = "think",
        has_learned_path: bool = False,
    ) -> WorkspaceState:
        layers = brain.layer_activation_summary()
        sensory = float(layers.get("sensory", 0.0))
        assoc = float(layers.get("associative", 0.0))
        motor = float(layers.get("motor", 0.0))

        ignition = (
            0.28 * sensory
            + 0.34 * assoc
            + 0.14 * motor
            + 0.26 * thought.pressure
            + 0.12 * novelty
            + (0.06 if has_learned_path else 0.0)
        )
        ignition = min(1.0, ignition)
        conscious = ignition >= self.threshold

        self_signal = min(1.0, 0.4 * assoc + 0.35 * thought.pressure + 0.15 * self._self_continuity)
        if conscious:
            self._self_continuity = min(1.0, self._self_continuity * 0.92 + self_signal * 0.12)

        broadcast: list[str] = list(thought.themes[:6])
        for sym in (sensory_symbols or [])[:4]:
            if sym not in broadcast:
                broadcast.append(sym)
        focus = broadcast[0] if broadcast else "FOCUS:active"
        if conscious:
            self._ignitions += 1
            self._ignite(brain, ignition)
            self._last_focus = focus

        mode = self._decide_mode(
            ignition=ignition,
            assoc=assoc,
            motor=motor,
            wave_phase=wave_phase,
            thought_pressure=thought.pressure,
            conscious=conscious,
        )

        return WorkspaceState(
            conscious=conscious,
            ignition=ignition,
            broadcast=broadcast,
            focus=focus if conscious else (self._last_focus or "FOCUS:sub"),
            mode=mode,
            self_signal=self_signal,
        )

    def _decide_mode(
        self,
        *,
        ignition: float,
        assoc: float,
        motor: float,
        wave_phase: str,
        thought_pressure: float,
        conscious: bool,
    ) -> OutputMode:
        """Flusso unico — parlare e riflettere non si escludono."""
        if not conscious and thought_pressure < 0.08:
            return "flow"
        if wave_phase == "reflect" and thought_pressure > 0.15:
            return "reflect"
        if motor > 0.05 or thought_pressure > 0.12 or ignition > 0.3:
            return "speak"
        if assoc > 0.1 and thought_pressure > 0.08:
            return "flow"
        return "speak"

    def _ignite(self, brain: NeuralTopology, intensity: float) -> None:
        for n in brain.get_neurons("associative", "pattern_matcher"):
            n.activation = min(1.0, n.activation + intensity * 0.2)
        for n in brain.get_neurons("associative", "memory_consolidator"):
            n.activation = min(1.0, n.activation + intensity * 0.1)
        brain.propagate(steps=1)
        brain.reinforce_active_pathway(boost=0.03 * intensity)

    def stats(self) -> dict[str, Any]:
        return {
            "ignitions": self._ignitions,
            "threshold": self.threshold,
            "last_focus": self._last_focus,
            "self_continuity": round(self._self_continuity, 3),
        }
