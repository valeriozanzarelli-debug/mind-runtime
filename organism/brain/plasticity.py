"""Hebbian, STDP-ish, and homeostatic plasticity rules — sparse at scale."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from organism.brain.topology import NeuralTopology


class PlasticityEngine:
    def __init__(self, config: dict) -> None:
        self.hebbian = config.get("hebbian_learning", {})
        self.stdp = config.get("spike_timing_dependent", {})
        self.homeostatic = config.get("homeostatic", {})

    def apply_hebbian(self, brain: NeuralTopology, t: float) -> int:  # noqa: ARG002
        """Neurons that fire together, wire together — solo bordi da presinaptici attivi."""
        rate = float(self.hebbian.get("rate", 0.01))
        decay = float(self.hebbian.get("decay", 0.001))
        updated = 0
        scan = brain._active if brain._active else brain.neurons.keys()
        for nid in scan:
            pre = brain.neurons[nid]
            if pre.activation <= 0.3:
                continue
            for syn in brain.outgoing.get(nid, ()):
                post = brain.neurons[syn.post_id]
                if post.activation > 0.3:
                    syn.weight = min(1.0, syn.weight + rate * pre.activation * post.activation)
                    updated += 1
                else:
                    syn.weight = max(0.0, syn.weight - decay)
        return updated

    def apply_stdp(self, brain: NeuralTopology, t: float) -> int:  # noqa: ARG002
        window = float(self.stdp.get("window_ms", 20)) / 1000.0
        pot = float(self.stdp.get("potentiation", 0.05))
        dep = float(self.stdp.get("depression", 0.02))
        updated = 0
        scan = brain._active if brain._active else brain.neurons.keys()
        for nid in scan:
            pre = brain.neurons[nid]
            if pre.last_spike_t < 0:
                continue
            for syn in brain.outgoing.get(nid, ()):
                post = brain.neurons[syn.post_id]
                if post.last_spike_t < 0:
                    continue
                dt = post.last_spike_t - pre.last_spike_t
                if 0 < dt <= window:
                    syn.weight = min(1.0, syn.weight + pot)
                    updated += 1
                elif -window <= dt < 0:
                    syn.weight = max(0.0, syn.weight - dep)
                    updated += 1
        return updated

    def homeostatic_tick(self, brain: NeuralTopology) -> None:
        target = float(self.homeostatic.get("target_firing_rate", 5.0))
        adj = float(self.homeostatic.get("adjustment_rate", 0.001))
        scan = brain._active if brain._active else brain.neurons.values()
        for n in scan:
            if isinstance(n, int):
                n = brain.neurons[n]
            if n.spike_count > target:
                for syn in brain.outgoing.get(n.id, []):
                    syn.weight = max(0.0, syn.weight - adj)
            elif n.spike_count < target * 0.2 and n.spike_count > 0:
                for syn in brain.outgoing.get(n.id, []):
                    syn.weight = min(1.0, syn.weight + adj)
