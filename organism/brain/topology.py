"""Neural topology — sparse graph, spike propagation, region indexing."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from organism.brain.neuron import Neuron
from organism.brain.plasticity import PlasticityEngine
from organism.brain.synapse import Synapse

FIRE_THRESHOLD = 0.35
ACTIVE_FLOOR = 0.01


@dataclass
class Spike:
    neuron_id: int
    timestamp: float
    intensity: float = 1.0


@dataclass
class ActivePattern:
    id: str
    modality: str
    strength: float
    neuron_ids: list[int] = field(default_factory=list)


class NeuralTopology:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.neurons: dict[int, Neuron] = {}
        self.synapses: list[Synapse] = []
        self.outgoing: dict[int, list[Synapse]] = defaultdict(list)
        self.incoming: dict[int, list[Synapse]] = defaultdict(list)
        self._edge_keys: set[tuple[int, int]] = set()
        self._active: set[int] = set()
        self._next_id = 0
        self.tick: float = 0.0
        self.plasticity: PlasticityEngine | None = None
        self._by_region: dict[str, list[int]] = defaultdict(list)
        self._by_system: dict[str, list[int]] = defaultdict(list)
        self.energy_budget: int = 0

    @property
    def neuron_count(self) -> int:
        return len(self.neurons)

    @property
    def synapse_count(self) -> int:
        return len(self.synapses)

    @property
    def active_count(self) -> int:
        return len(self._active)

    def add_neuron(self, system: str, region: str, meta: dict[str, Any] | None = None) -> Neuron:
        nid = self._next_id
        self._next_id += 1
        n = Neuron(id=nid, system=system, region=region, meta=meta or {})
        self.neurons[nid] = n
        self._by_region[region].append(nid)
        self._by_system[system].append(nid)
        return n

    def add_neurons_bulk(
        self,
        system: str,
        region: str,
        count: int,
        *,
        meta_fn: Any | None = None,
    ) -> list[int]:
        ids: list[int] = []
        bucket = self._by_region[region]
        for i in range(count):
            nid = self._next_id
            self._next_id += 1
            meta: dict[str, Any] = meta_fn(i, count) if meta_fn else {"index": i, "region": region}
            self.neurons[nid] = Neuron(id=nid, system=system, region=region, meta=meta)
            bucket.append(nid)
            self._by_system[system].append(nid)
            ids.append(nid)
        return ids

    def region_ids(self, region: str) -> list[int]:
        return list(self._by_region.get(region, []))

    def system_ids(self, system: str) -> list[int]:
        return list(self._by_system.get(system, []))

    def region_activation(self, region: str) -> float:
        ids = self._by_region.get(region, [])
        if not ids:
            return 0.0
        return sum(self.neurons[n].activation for n in ids) / len(ids)

    def all_region_activations(self) -> dict[str, float]:
        return {r: self.region_activation(r) for r in self._by_region}

    def connect(
        self,
        pre_id: int,
        post_id: int,
        weight: float | None = None,
        *,
        weight_init: str = "xavier_uniform",
        pathway: str = "",
        plastic: bool = True,
        dopamine_modulated: bool = False,
    ) -> Synapse:
        if weight is None:
            weight = self._init_weight(weight_init)
        syn = Synapse(
            pre_id=pre_id,
            post_id=post_id,
            weight=weight,
            pathway=pathway,
            plastic=plastic,
            dopamine_modulated=dopamine_modulated,
        )
        self.synapses.append(syn)
        self.outgoing[pre_id].append(syn)
        self.incoming[post_id].append(syn)
        self._edge_keys.add((pre_id, post_id))
        return syn

    def connect_regions(
        self,
        source_region: str,
        target_region: str,
        *,
        connections_per_neuron: int = 20,
        weight_init: str = "xavier_uniform",
        pathway: str = "",
        plastic: bool = True,
        dopamine_modulated: bool = False,
    ) -> int:
        sources = self.region_ids(source_region)
        targets = self.region_ids(target_region)
        if not sources or not targets:
            return 0
        k = min(connections_per_neuron, len(targets))
        added = 0
        for sid in sources:
            picks = self._sample(targets, k)
            for tid in picks:
                if (sid, tid) in self._edge_keys:
                    continue
                self.connect(
                    sid,
                    tid,
                    weight_init=weight_init,
                    pathway=pathway,
                    plastic=plastic,
                    dopamine_modulated=dopamine_modulated,
                )
                added += 1
        return added

    def set_plasticity(self, config: dict) -> None:
        self.plasticity = PlasticityEngine(config)

    def inject_spikes(self, spikes: list[Spike]) -> None:
        for sp in spikes:
            if sp.neuron_id in self.neurons:
                self.neurons[sp.neuron_id].fire(sp.timestamp, sp.intensity)
                if sp.intensity > ACTIVE_FLOOR:
                    self._active.add(sp.neuron_id)

    def inject_region(self, region: str, intensity: float, t: float, *, fraction: float = 0.3) -> int:
        """Activate a fraction of neurons in a region (sensory input)."""
        ids = self.region_ids(region)
        if not ids:
            return 0
        k = max(1, int(len(ids) * fraction))
        picks = self._sample(ids, k)
        for nid in picks:
            self.neurons[nid].fire(t, intensity)
            self._active.add(nid)
        return len(picks)

    def active_neurons(self) -> set[int]:
        return self._active

    def propagate(self, dt: float = 0.001) -> dict[str, Any]:
        """One propagation step — sparse, energy-budget aware."""
        self.tick += dt
        t = self.tick
        fired: list[int] = []
        budget = self.energy_budget or len(self._active) * 200
        spent = 0

        to_fire: dict[int, float] = defaultdict(float)
        scan = list(self._active) if self._active else []

        for nid in scan:
            if spent >= budget:
                break
            pre = self.neurons.get(nid)
            if pre is None or pre.activation < FIRE_THRESHOLD:
                continue
            for syn in self.outgoing.get(nid, ()):
                if spent >= budget:
                    break
                signal = syn.transmit(pre.activation)
                if signal > ACTIVE_FLOOR:
                    to_fire[syn.post_id] += signal
                    spent += 1

        new_active: set[int] = set()
        for nid, signal in to_fire.items():
            n = self.neurons.get(nid)
            if n is None:
                continue
            n.activation = min(1.0, n.activation + signal)
            if n.activation >= FIRE_THRESHOLD:
                n.fire(t, n.activation)
                fired.append(nid)
                new_active.add(nid)
            elif n.activation > ACTIVE_FLOOR:
                new_active.add(nid)

        decayed_off: set[int] = set()
        for nid in list(self._active):
            n = self.neurons[nid]
            if not n.leak():
                decayed_off.add(nid)
        self._active = (self._active - decayed_off) | new_active

        return {"fired": len(fired), "active": len(self._active), "tick": t}

    def plasticity_tick(self, dopamine_level: float = 0.5) -> dict[str, int]:
        if self.plasticity is None:
            return {"hebbian": 0, "stdp": 0}
        h = self.plasticity.apply_hebbian(self, self.tick, dopamine_level)
        s = self.plasticity.apply_stdp(self, self.tick, dopamine_level)
        return {"hebbian": h, "stdp": s}

    def _init_weight(self, mode: str) -> float:
        if mode == "xavier_uniform":
            return self.rng.uniform(0.02, 0.15)
        if mode == "small_random":
            return self.rng.uniform(0.01, 0.08)
        return self.rng.uniform(0.05, 0.2)

    def _sample(self, items: list[int], k: int) -> list[int]:
        if k >= len(items):
            return list(items)
        return self.rng.sample(items, k)

    def stats(self) -> dict[str, Any]:
        by_region = {r: len(ids) for r, ids in self._by_region.items()}
        return {
            "neurons": self.neuron_count,
            "synapses": self.synapse_count,
            "active": self.active_count,
            "tick": self.tick,
            "regions": by_region,
        }

    def to_csr_arrays(self) -> tuple[Any, Any, Any]:
        """Export synapses as CSR (row_ptr, col_idx, weights) for GPU."""
        import numpy as np

        n = self.neuron_count
        if not self.synapses:
            return (
                np.zeros(n + 1, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
            )
        rows, cols, weights = [], [], []
        for syn in self.synapses:
            rows.append(syn.pre_id)
            cols.append(syn.post_id)
            weights.append(syn.weight)
        order = np.argsort(rows)
        rows = np.array(rows, dtype=np.int32)[order]
        cols = np.array(cols, dtype=np.int32)[order]
        weights = np.array(weights, dtype=np.float32)[order]
        row_ptr = np.zeros(n + 1, dtype=np.int32)
        for r in rows:
            row_ptr[r + 1] += 1
        row_ptr = np.cumsum(row_ptr)
        return row_ptr, cols, weights
