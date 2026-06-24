"""Hebbian, STDP, and dopamine-modulated plasticity."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from organism.brain.topology import NeuralTopology


class PlasticityEngine:
    def __init__(self, config: dict) -> None:
        self.hebbian = config.get("hebbian_learning", {})
        self.stdp = config.get("spike_timing_dependent", {})
        self.homeostatic = config.get("homeostatic", {})

    def apply_hebbian(self, brain: NeuralTopology, t: float, dopamine_level: float = 0.5) -> int:  # noqa: ARG002
        rate = float(self.hebbian.get("rate", 0.01))
        decay = float(self.hebbian.get("decay", 0.001))
        mod = 0.5 + dopamine_level
        updated = 0
        for nid in brain.active_neurons():
            pre = brain.neurons[nid]
            if pre.activation <= 0.3:
                continue
            for syn in brain.outgoing.get(nid, ()):
                if not syn.plastic:
                    continue
                post = brain.neurons[syn.post_id]
                if post.activation > 0.3:
                    boost = rate * mod if syn.dopamine_modulated else rate
                    syn.weight = min(1.0, syn.weight + boost * pre.activation * post.activation)
                    updated += 1
                else:
                    syn.weight = max(0.0, syn.weight - decay)
        return updated

    def apply_stdp(self, brain: NeuralTopology, t: float, dopamine_level: float = 0.5) -> int:
        window = float(self.stdp.get("window_ms", 20)) / 1000.0
        pot = float(self.stdp.get("potentiation", 0.05))
        dep = float(self.stdp.get("depression", 0.02))
        mod = 0.5 + dopamine_level if dopamine_level > 0.2 else 1.0
        updated = 0
        for nid in brain.active_neurons():
            pre = brain.neurons[nid]
            if pre.last_spike_t < 0:
                continue
            for syn in brain.outgoing.get(nid, ()):
                if not syn.plastic:
                    continue
                post = brain.neurons[syn.post_id]
                if post.last_spike_t < 0:
                    continue
                dt = post.last_spike_t - pre.last_spike_t
                if 0 < dt <= window:
                    delta = pot * mod if syn.dopamine_modulated else pot
                    syn.weight = min(1.0, syn.weight + delta)
                    updated += 1
                elif -window <= dt < 0:
                    syn.weight = max(0.0, syn.weight - dep)
                    updated += 1
        return updated

    def homeostatic_tick(self, brain: NeuralTopology) -> None:
        target = float(self.homeostatic.get("target_firing_rate", 5.0))
        adj = float(self.homeostatic.get("adjustment_rate", 0.001))
        for n in brain.neurons.values():
            if n.spike_count > target:
                for syn in brain.outgoing.get(n.id, []):
                    syn.weight = max(0.0, syn.weight - adj)
            elif 0 < n.spike_count < target * 0.2:
                for syn in brain.outgoing.get(n.id, []):
                    syn.weight = min(1.0, syn.weight + adj)
