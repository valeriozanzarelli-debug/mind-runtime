"""Neuroni compatti — array numpy invece di oggetti Python (~48 B vs ~2.65 KB).

Abilita decine/centinaia di milioni di neuroni grafo sulla RAM del server.
I neuroni «interfaccia» (sensory/motor) restano oggetti Python per compatibilità.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False

LAYER_ENC = {"sensory": 0, "associative": 1, "motor": 2}


def compact_enabled() -> bool:
    return os.environ.get("ORGANISM_COMPACT_BRAIN", "0") not in ("0", "false", "no")


def estimate_bytes_per_neuron() -> int:
    return 48


class CompactNeuralBackend:
    """Bulk associative neurons — propagazione su array numpy."""

    def __init__(self) -> None:
        if not HAS_NUMPY:
            raise RuntimeError("numpy richiesto per CompactNeuralBackend")
        self.ids: np.ndarray = np.empty(0, dtype=np.int32)
        self.activation: np.ndarray = np.empty(0, dtype=np.float32)
        self.layer: np.ndarray = np.empty(0, dtype=np.uint8)
        self.subtype_id: np.ndarray = np.empty(0, dtype=np.uint16)
        self.subtype_names: list[str] = []
        self._subtype_index: dict[str, int] = {}
        self.pre: np.ndarray = np.empty(0, dtype=np.int32)
        self.post: np.ndarray = np.empty(0, dtype=np.int32)
        self.weight: np.ndarray = np.empty(0, dtype=np.float32)
        self._id_to_local: dict[int, int] = {}

    @property
    def count(self) -> int:
        return len(self.ids)

    def subtype_index(self, name: str) -> int:
        if name not in self._subtype_index:
            self._subtype_index[name] = len(self.subtype_names)
            self.subtype_names.append(name)
        return self._subtype_index[name]

    def add_neurons(self, start_id: int, layer: str, subtype: str, count: int) -> list[int]:
        if count <= 0:
            return []
        layer_code = LAYER_ENC.get(layer, 1)
        st_id = self.subtype_index(subtype)
        new_ids = np.arange(start_id, start_id + count, dtype=np.int32)
        self.ids = np.concatenate([self.ids, new_ids])
        self.activation = np.concatenate([self.activation, np.zeros(count, dtype=np.float32)])
        self.layer = np.concatenate([self.layer, np.full(count, layer_code, dtype=np.uint8)])
        self.subtype_id = np.concatenate([self.subtype_id, np.full(count, st_id, dtype=np.uint16)])
        base = len(self.ids) - count
        for i, nid in enumerate(new_ids):
            self._id_to_local[int(nid)] = base + i
        return new_ids.tolist()

    def add_edges(self, edges: list[tuple[int, int, float]]) -> int:
        if not edges:
            return 0
        self.pre = np.concatenate([self.pre, np.array([e[0] for e in edges], dtype=np.int32)])
        self.post = np.concatenate([self.post, np.array([e[1] for e in edges], dtype=np.int32)])
        self.weight = np.concatenate([self.weight, np.array([e[2] for e in edges], dtype=np.float32)])
        return len(edges)

    def set_activation(self, neuron_id: int, value: float) -> None:
        loc = self._id_to_local.get(neuron_id)
        if loc is not None:
            self.activation[loc] = float(value)

    def get_activation(self, neuron_id: int) -> float:
        loc = self._id_to_local.get(neuron_id)
        return float(self.activation[loc]) if loc is not None else 0.0

    def leak(self, decay: float, floor: float = 0.01) -> None:
        self.activation = np.maximum(floor, self.activation * (1.0 - decay))

    def propagate_from_python(
        self,
        python_activation: dict[int, float],
        python_outgoing: dict[int, list[Any]],
        *,
        spread_thresh: float = 0.05,
        budget: int = 0,
    ) -> dict[int, float]:
        delta_compact = np.zeros(len(self.ids), dtype=np.float32)
        delta_py: dict[int, float] = {}
        processed = 0

        for pre_id, act in python_activation.items():
            if act <= spread_thresh:
                continue
            for syn in python_outgoing.get(pre_id, ()):
                loc = self._id_to_local.get(syn.post_id)
                if loc is not None:
                    delta_compact[loc] += syn.weight * act
                processed += 1
                if budget and processed >= budget:
                    break
            if budget and processed >= budget:
                break

        active_loc = np.where(self.activation > spread_thresh)[0]
        for loc in active_loc:
            act = float(self.activation[loc])
            nid = int(self.ids[loc])
            for syn in python_outgoing.get(nid, ()):
                post_loc = self._id_to_local.get(syn.post_id)
                if post_loc is not None:
                    delta_compact[post_loc] += syn.weight * act
                else:
                    delta_py[syn.post_id] = delta_py.get(syn.post_id, 0.0) + syn.weight * act

        for i in range(len(self.pre)):
            pre_loc = self._id_to_local.get(int(self.pre[i]))
            if pre_loc is None or self.activation[pre_loc] <= spread_thresh:
                continue
            act = float(self.activation[pre_loc])
            post = int(self.post[i])
            w = float(self.weight[i])
            post_loc = self._id_to_local.get(post)
            if post_loc is not None:
                delta_compact[post_loc] += w * act
            else:
                delta_py[post] = delta_py.get(post, 0.0) + w * act

        self.activation = np.minimum(1.0, self.activation + delta_compact * 0.35)
        return delta_py

    def active_ids(self, threshold: float = 0.05) -> list[int]:
        return self.ids[self.activation > threshold].tolist()

    def stats(self) -> dict[str, Any]:
        return {
            "compact_neurons": self.count,
            "compact_synapses": len(self.pre),
            "bytes_per_neuron": estimate_bytes_per_neuron(),
            "estimated_mb": round(self.count * estimate_bytes_per_neuron() / (1024 * 1024), 2),
        }
