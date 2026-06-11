"""Neural topology — sparse graph, spike propagation, pruning, visualization."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from organism.brain.neuron import Neuron
from organism.brain.plasticity import PlasticityEngine
from organism.brain.synapse import Synapse

FIRE_THRESHOLD = 0.05
ACTIVE_FLOOR = 0.01
LARGE_BRAIN_THRESHOLD = 50_000


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
        self.tick = 0.0
        self.plasticity: PlasticityEngine | None = None
        self._by_layer_subtype: dict[tuple[str, str], list[int]] = defaultdict(list)
        # 0 = illimitato; a scala mega limita fan-out per tick (metabolismo)
        self.energy_budget: int = 0
        self.spreading_activation_threshold: float = 0.3

    @property
    def neuron_count(self) -> int:
        return len(self.neurons)

    @property
    def synapse_count(self) -> int:
        return len(self.synapses)

    @property
    def active_count(self) -> int:
        return len(self._active)

    def add_neuron(self, layer: str, subtype: str, meta: dict[str, Any] | None = None) -> Neuron:
        nid = self._next_id
        self._next_id += 1
        n = Neuron(id=nid, layer=layer, subtype=subtype, meta=meta or {})
        self.neurons[nid] = n
        self._by_layer_subtype[(layer, subtype)].append(nid)
        return n

    def add_neurons_bulk(
        self,
        layer: str,
        subtype: str,
        count: int,
        *,
        meta_fn: Any | None = None,
    ) -> list[int]:
        """Creazione rapida — milioni di neuroni senza loop Python pesante per meta."""
        ids: list[int] = []
        bucket = self._by_layer_subtype[(layer, subtype)]
        for i in range(count):
            nid = self._next_id
            self._next_id += 1
            meta: dict[str, Any] = meta_fn(i, count) if meta_fn else {"index": i}
            self.neurons[nid] = Neuron(id=nid, layer=layer, subtype=subtype, meta=meta)
            bucket.append(nid)
            ids.append(nid)
        return ids

    def connect(
        self,
        pre_id: int,
        post_id: int,
        weight: float | None = None,
        *,
        weight_init: str = "xavier_uniform",
    ) -> Synapse:
        if weight is None:
            weight = self._init_weight(weight_init)
        syn = Synapse(pre_id=pre_id, post_id=post_id, weight=weight)
        self.synapses.append(syn)
        self.outgoing[pre_id].append(syn)
        self.incoming[post_id].append(syn)
        self._edge_keys.add((pre_id, post_id))
        return syn

    def has_edge(self, pre_id: int, post_id: int) -> bool:
        return (pre_id, post_id) in self._edge_keys

    def connect_layers(
        self,
        source_layer: str,
        target_layer: str,
        *,
        source_subtype: str | None = None,
        target_subtype: str | None = None,
        connections_per_neuron: int = 50,
        weight_init: str = "xavier_uniform",
        pruning_threshold: float = 0.0,
    ) -> int:
        sources = self._filter_ids(source_layer, source_subtype)
        targets = self._filter_ids(target_layer, target_subtype)
        if not sources or not targets:
            return 0
        k = min(connections_per_neuron, len(targets))
        batch: list[Synapse] = []
        for sid in sources:
            picks = self._sample(targets, k)
            for tid in picks:
                if (sid, tid) in self._edge_keys:
                    continue
                w = self._init_weight(weight_init)
                if w < pruning_threshold:
                    continue
                syn = Synapse(pre_id=sid, post_id=tid, weight=w)
                batch.append(syn)
                self._edge_keys.add((sid, tid))
        for syn in batch:
            self.synapses.append(syn)
            self.outgoing[syn.pre_id].append(syn)
            self.incoming[syn.post_id].append(syn)
        return len(batch)

    def set_plasticity(self, config: dict) -> None:
        self.plasticity = PlasticityEngine(config)

    def inject_spikes(self, spikes: list[Spike]) -> None:
        for sp in spikes:
            if sp.neuron_id in self.neurons:
                self.neurons[sp.neuron_id].fire(sp.timestamp, sp.intensity)
                if sp.intensity > ACTIVE_FLOOR:
                    self._active.add(sp.neuron_id)

    def propagate(self, steps: int = 2, decay: float = 0.12) -> None:
        """Diffusione sparsa — solo neuroni attivi propagano (come corteccia reale)."""
        self.tick += 1.0
        for _ in range(steps):
            if not self._active:
                if self.neuron_count < LARGE_BRAIN_THRESHOLD:
                    self._collect_active(FIRE_THRESHOLD)
                if not self._active:
                    break

            delta: dict[int, float] = defaultdict(float)
            budget = self.energy_budget
            processed = 0
            # Prima i più attivi — priorità metabolica
            spread_thresh = max(FIRE_THRESHOLD, self.spreading_activation_threshold)
            firing = sorted(
                (nid for nid in self._active if self.neurons[nid].activation > spread_thresh),
                key=lambda i: self.neurons[i].activation,
                reverse=True,
            )
            for nid in firing:
                pre = self.neurons[nid]
                for syn in self.outgoing.get(nid, ()):
                    delta[syn.post_id] += syn.transmit(pre.activation)
                    processed += 1
                    if budget and processed >= budget:
                        break
                if budget and processed >= budget:
                    break

            touched: set[int] = set(firing)
            next_active: set[int] = set()
            for nid, act in delta.items():
                n = self.neurons[nid]
                n.activation = min(1.0, n.activation + act * 0.35)
                touched.add(nid)
                if n.activation > FIRE_THRESHOLD:
                    next_active.add(nid)

            for nid in touched:
                if self.neurons[nid].leak(decay, floor=ACTIVE_FLOOR):
                    if self.neurons[nid].activation > FIRE_THRESHOLD:
                        next_active.add(nid)

            for nid in self._active - touched:
                n = self.neurons[nid]
                if n.leak(decay * 0.5, floor=ACTIVE_FLOOR) and n.activation > FIRE_THRESHOLD:
                    next_active.add(nid)

            self._active = next_active

    def get_neurons(self, layer: str, subtype: str) -> list[Neuron]:
        return [self.neurons[i] for i in self._by_layer_subtype.get((layer, subtype), [])]

    def get_active_patterns(
        self,
        threshold: float = 0.45,
        modality: str | None = None,
    ) -> list[ActivePattern]:
        """Cluster highly active associative neurons into named patterns."""
        patterns: list[ActivePattern] = []
        for (layer, subtype), ids in self._by_layer_subtype.items():
            if layer != "associative":
                continue
            active = [i for i in ids if self.neurons[i].activation >= threshold]
            if not active:
                continue
            strength = sum(self.neurons[i].activation for i in active) / len(active)
            mod = subtype.split("_")[0] if "_" in subtype else "general"
            if modality and mod != modality and modality not in subtype:
                continue
            patterns.append(
                ActivePattern(
                    id=f"{subtype}_active",
                    modality=mod,
                    strength=strength,
                    neuron_ids=active[:20],
                )
            )
        return sorted(patterns, key=lambda p: p.strength, reverse=True)

    def prune_weak_synapses(self, threshold: float = 0.05, keep_percentage: float = 0.85) -> int:
        if not self.synapses:
            return 0
        self.synapses.sort(key=lambda s: s.weight)
        cut = int(len(self.synapses) * (1.0 - keep_percentage))
        to_remove = {i for i, s in enumerate(self.synapses) if s.weight < threshold}
        to_remove.update(range(cut))
        kept = [s for i, s in enumerate(self.synapses) if i not in to_remove]
        removed = len(self.synapses) - len(kept)
        self.synapses = kept
        self._rebuild_adjacency()
        return removed

    def active_neuron_ids(self, threshold: float = 0.25) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        scan = self._active if self._active else self.neurons
        for nid in scan:
            n = self.neurons[nid]
            if n.activation >= threshold:
                out[n.layer].append(nid)
        return dict(out)

    def reinforce_active_pathway(self, boost: float = 0.03, threshold: float = 0.2) -> int:
        """Rinforzo sparsa — solo sinapsi da neuroni attivi."""
        active = self.active_neuron_ids(threshold)
        sensory = set(active.get("sensory", []))
        assoc = set(active.get("associative", []))
        motor = set(active.get("motor", []))
        if not assoc and not sensory:
            return 0
        strengthened = 0
        seen: set[int] = set()
        for nid in sensory | assoc:
            if nid in seen:
                continue
            seen.add(nid)
            pre_n = self.neurons[nid]
            for syn in self.outgoing.get(nid, ()):
                post_n = self.neurons[syn.post_id]
                on_path = False
                if pre_n.layer == "sensory" and post_n.layer == "associative":
                    on_path = nid in sensory or syn.post_id in assoc
                elif pre_n.layer == "associative" and post_n.layer == "associative":
                    on_path = nid in assoc or syn.post_id in assoc
                elif pre_n.layer == "associative" and post_n.layer == "motor":
                    on_path = nid in assoc or syn.post_id in motor
                elif pre_n.layer == "sensory" and post_n.layer == "motor":
                    on_path = nid in sensory
                if on_path:
                    syn.weight = min(1.0, syn.weight + boost * max(pre_n.activation, 0.1))
                    strengthened += 1
        return strengthened

    def mean_synapse_weight(self) -> float:
        if not self.synapses:
            return 0.0
        return sum(s.weight for s in self.synapses) / len(self.synapses)

    def consolidate_memory(self) -> dict[str, int]:
        """Sleep tick — plasticity + homeostasis."""
        stats = {"hebbian": 0, "stdp": 0}
        if self.plasticity:
            stats["hebbian"] = self.plasticity.apply_hebbian(self, self.tick)
            stats["stdp"] = self.plasticity.apply_stdp(self, self.tick)
            self.plasticity.homeostatic_tick(self)
        for n in self.neurons.values():
            n.spike_count = max(0, n.spike_count // 2)
        return stats

    def visualize(self, max_nodes: int = 40) -> dict[str, Any]:
        """JSON graph summary for debugging."""
        nodes = []
        for nid, n in list(self.neurons.items())[:max_nodes]:
            nodes.append(
                {
                    "id": nid,
                    "layer": n.layer,
                    "subtype": n.subtype,
                    "activation": round(n.activation, 3),
                }
            )
        edges = [
            {"from": s.pre_id, "to": s.post_id, "weight": round(s.weight, 3)}
            for s in self.synapses[: min(200, len(self.synapses))]
        ]
        return {
            "neurons": self.neuron_count,
            "synapses": self.synapse_count,
            "active": self.active_count,
            "nodes_sample": nodes,
            "edges_sample": edges,
        }

    def visualize_json(self, **kwargs) -> str:
        return json.dumps(self.visualize(**kwargs), indent=2)

    def export_active_subgraph(
        self,
        *,
        activation_threshold: float = 0.15,
        max_neurons: int = 80,
        max_edges: int = 300,
    ) -> dict[str, Any]:
        """Subgraph of currently active neurons + connecting synapses — for nursery viz."""
        active_ids = [
            nid for nid, n in self.neurons.items() if n.activation >= activation_threshold
        ]
        active_ids.sort(key=lambda i: self.neurons[i].activation, reverse=True)
        active_ids = active_ids[:max_neurons]
        active_set = set(active_ids)

        if not active_set:
            degree: dict[int, int] = defaultdict(int)
            for s in self.synapses[:5000]:
                degree[s.pre_id] += 1
                degree[s.post_id] += 1
            active_ids = sorted(degree, key=degree.get, reverse=True)[: min(30, max_neurons)]
            active_set = set(active_ids)

        nodes = []
        for nid in active_set:
            n = self.neurons[nid]
            nodes.append(
                {
                    "id": nid,
                    "label": f"{n.subtype[:12]}",
                    "layer": n.layer,
                    "subtype": n.subtype,
                    "activation": round(n.activation, 4),
                    "spikes": n.spike_count,
                    "group": n.layer,
                }
            )

        edges = []
        for syn in self.synapses:
            if syn.pre_id in active_set and syn.post_id in active_set:
                edges.append(
                    {
                        "from": syn.pre_id,
                        "to": syn.post_id,
                        "weight": round(syn.weight, 4),
                        "width": max(1, syn.weight * 4),
                    }
                )
                if len(edges) >= max_edges:
                    break

        layer_counts = defaultdict(int)
        for nid in active_set:
            layer_counts[self.neurons[nid].layer] += 1

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "total_neurons": self.neuron_count,
                "total_synapses": self.synapse_count,
                "active_shown": len(nodes),
                "edges_shown": len(edges),
                "mean_weight": round(self.mean_synapse_weight(), 5),
                "tick": self.tick,
                "layer_active": dict(layer_counts),
            },
        }

    def layer_activation_summary(self) -> dict[str, float]:
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        scan = self._active if len(self._active) < self.neuron_count // 2 else self.neurons
        for nid in scan:
            n = self.neurons[nid]
            sums[n.layer] += n.activation
            counts[n.layer] += 1
        return {layer: round(sums[layer] / max(1, counts[layer]), 4) for layer in sums}

    def efficiency_stats(self) -> dict[str, Any]:
        """Metriche metaboliche — quanto del cervello è acceso per tick."""
        n = max(1, self.neuron_count)
        s = max(1, self.synapse_count)
        active = self.active_count
        fan = s / n
        return {
            "neurons": n,
            "synapses": s,
            "active_neurons": active,
            "active_ratio": round(active / n, 6),
            "mean_fan_out": round(fan, 2),
            "energy_budget": self.energy_budget,
            "sparse": True,
        }

    def _collect_active(self, threshold: float) -> None:
        self._active = {nid for nid, n in self.neurons.items() if n.activation > threshold}

    def _filter_ids(self, layer: str, subtype: str | None) -> list[int]:
        if subtype:
            return list(self._by_layer_subtype.get((layer, subtype), []))
        return [i for (l, _), ids in self._by_layer_subtype.items() if l == layer for i in ids]

    def _sample(self, items: list[int], k: int) -> list[int]:
        if k >= len(items):
            return list(items)
        return self.rng.sample(items, k)

    def _init_weight(self, scheme: str) -> float:
        if scheme == "xavier_uniform":
            return self.rng.uniform(0.05, 0.35)
        if scheme == "small_random":
            return self.rng.uniform(0.01, 0.15)
        return self.rng.random() * 0.5

    def snapshot(self) -> dict[str, Any]:
        """Topologia completa — neuroni aggiunti + sinapsi dinamiche."""
        return {
            "tick": self.tick,
            "next_id": self._next_id,
            "neurons": [
                {"id": n.id, "layer": n.layer, "subtype": n.subtype, "meta": dict(n.meta)}
                for n in self.neurons.values()
            ],
            "synapses": [
                {"pre": s.pre_id, "post": s.post_id, "w": s.weight} for s in self.synapses
            ],
        }

    def restore_snapshot(self, data: dict[str, Any]) -> dict[str, int]:
        """Ripristina neuroni e sinapsi salvati — non solo i pesi DNA."""
        added_neurons = 0
        added_synapses = 0
        updated_weights = 0
        self.tick = float(data.get("tick", self.tick))
        self._next_id = max(self._next_id, int(data.get("next_id", self._next_id)))

        for row in data.get("neurons", []):
            nid = int(row["id"])
            if nid in self.neurons:
                continue
            layer = str(row["layer"])
            subtype = str(row["subtype"])
            meta = dict(row.get("meta", {}))
            self.neurons[nid] = Neuron(id=nid, layer=layer, subtype=subtype, meta=meta)
            self._by_layer_subtype[(layer, subtype)].append(nid)
            self._next_id = max(self._next_id, nid + 1)
            added_neurons += 1

        syn_map = {(int(s["pre"]), int(s["post"])): float(s["w"]) for s in data.get("synapses", [])}
        for key, weight in syn_map.items():
            pre_id, post_id = key
            if pre_id not in self.neurons or post_id not in self.neurons:
                continue
            if self.has_edge(pre_id, post_id):
                for syn in self.outgoing.get(pre_id, []):
                    if syn.post_id == post_id:
                        syn.weight = weight
                        updated_weights += 1
                        break
            else:
                self.connect(pre_id, post_id, weight=weight)
                added_synapses += 1
        return {
            "added_neurons": added_neurons,
            "added_synapses": added_synapses,
            "updated_weights": updated_weights,
        }

    def _rebuild_adjacency(self) -> None:
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        self._edge_keys = set()
        for syn in self.synapses:
            self.outgoing[syn.pre_id].append(syn)
            self.incoming[syn.post_id].append(syn)
            self._edge_keys.add((syn.pre_id, syn.post_id))
