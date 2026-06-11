"""Growth tracker — verifica che il DNA auto-sviluppi la struttura con l'apprendimento."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GrowthSnapshot:
    cycle: int
    phase: str
    neurons: int
    synapses: int
    mean_weight: float
    learning_cycles: int
    fragment_count: int
    learned_fragments: int
    layer_activation: dict[str, float]
    event: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "phase": self.phase,
            "neurons": self.neurons,
            "synapses": self.synapses,
            "mean_weight": self.mean_weight,
            "learning_cycles": self.learning_cycles,
            "fragment_count": self.fragment_count,
            "learned_fragments": self.learned_fragments,
            "layer_activation": self.layer_activation,
            "event": self.event,
        }


@dataclass
class GrowthVerification:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"auto_development": self.ok, "checks": self.checks}


class GrowthTracker:
    def __init__(self) -> None:
        self.timeline: list[GrowthSnapshot] = []

    def record(self, snap: GrowthSnapshot) -> None:
        self.timeline.append(snap)

    def timeline_dict(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.timeline]

    def verify_auto_development(self) -> GrowthVerification:
        """Checks that learning actually changes the organism over time."""
        checks: list[dict[str, Any]] = []
        ok = True

        if len(self.timeline) < 2:
            checks.append({"id": "min_snapshots", "ok": False, "detail": "servono almeno 2 snapshot"})
            return GrowthVerification(ok=False, checks=checks)

        birth = self.timeline[0]
        latest = self.timeline[-1]

        # DNA generated structure at birth
        checks.append(
            {
                "id": "dna_birth_structure",
                "ok": birth.neurons > 100 and birth.synapses > 1000,
                "detail": f"{birth.neurons} neuroni, {birth.synapses} sinapsi da DNA",
            }
        )

        # Mean synapse weight should grow with learning
        weight_grew = latest.mean_weight >= birth.mean_weight
        checks.append(
            {
                "id": "synapse_weight_growth",
                "ok": weight_grew,
                "detail": f"{birth.mean_weight:.5f} → {latest.mean_weight:.5f}",
            }
        )

        # Learning cycles advanced
        learned = latest.learning_cycles > birth.learning_cycles
        checks.append(
            {
                "id": "learning_cycles",
                "ok": learned,
                "detail": f"{birth.learning_cycles} → {latest.learning_cycles} cicli",
            }
        )

        # Learned fragments may appear
        frag_grew = latest.learned_fragments >= birth.learned_fragments
        checks.append(
            {
                "id": "memory_fragments",
                "ok": frag_grew,
                "detail": f"learned fragments {birth.learned_fragments} → {latest.learned_fragments}",
            }
        )

        # Neuron count stable (DNA topology) unless sleep pruned synapses only
        neurons_stable = latest.neurons == birth.neurons
        checks.append(
            {
                "id": "neuron_topology_stable",
                "ok": neurons_stable,
                "detail": f"neuroni {birth.neurons} → {latest.neurons} (DNA non aggiunge neuroni per ciclo)",
            }
        )

        for c in checks:
            if not c["ok"]:
                ok = False
        return GrowthVerification(ok=ok, checks=checks)
