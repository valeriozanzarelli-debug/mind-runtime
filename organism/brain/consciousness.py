"""Integrated Information (Φ) — dynamic consciousness metric."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConsciousnessState:
    phi: float = 0.0
    ignition: bool = False
    focus_region: str = ""
    complexity: float = 0.0
    integration: float = 0.0
    stream: list[str] = field(default_factory=list)
    _seq: int = 0

    def to_dict(self) -> dict:
        return {
            "phi": round(self.phi, 4),
            "ignition": self.ignition,
            "focus_region": self.focus_region,
            "complexity": round(self.complexity, 4),
            "integration": round(self.integration, 4),
        }


class PhiCalculator:
    """Approximate Φ from regional activation entropy and mutual information.

    Full IIT is intractable at 22k neurons; we use a practical proxy:
    Φ ≈ integration × complexity, where integration measures how much
    regions co-activate beyond independent expectation.
    """

    def __init__(self, ignition_threshold: float = 0.35) -> None:
        self.ignition_threshold = ignition_threshold

    def compute(
        self,
        region_activations: dict[str, float],
        task_complexity: float = 0.5,
    ) -> ConsciousnessState:
        if not region_activations:
            return ConsciousnessState()

        acts = np.array(list(region_activations.values()), dtype=np.float64)
        names = list(region_activations.keys())

        # Complexity: normalized entropy of activation distribution
        total = acts.sum()
        if total < 1e-9:
            probs = np.ones(len(acts)) / len(acts)
        else:
            probs = acts / total
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_entropy = np.log(len(acts))
        complexity = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        # Integration: variance of co-activation (high when multiple systems active together)
        active_mask = acts > 0.15
        n_active = int(active_mask.sum())
        if n_active < 2:
            integration = 0.0
        else:
            active_vals = acts[active_mask]
            integration = float(np.std(active_vals) / (np.mean(active_vals) + 1e-9))
            integration = min(1.0, integration)

        # Φ proxy scales with task complexity
        phi = complexity * integration * (0.5 + 0.5 * task_complexity)
        phi = min(1.0, phi * 1.2)

        focus_idx = int(np.argmax(acts))
        focus = names[focus_idx] if acts[focus_idx] > 0.1 else ""
        ignition = phi >= self.ignition_threshold

        state = ConsciousnessState(
            phi=phi,
            ignition=ignition,
            focus_region=focus,
            complexity=complexity,
            integration=integration,
        )
        return state

    def record_thought(self, state: ConsciousnessState, message: str) -> None:
        state._seq += 1
        prefix = "⚡" if state.ignition else "·"
        state.stream.append(f"{prefix} Φ={state.phi:.2f} [{state.focus_region}] {message}")
        if len(state.stream) > 100:
            state.stream = state.stream[-100:]
