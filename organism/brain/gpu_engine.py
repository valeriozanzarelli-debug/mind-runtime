"""GPU batch propagation — test neuron/synapse capacity on CUDA."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GPUCapacityResult:
    neurons: int
    synapses: int
    backend: str
    device: str
    propagate_ms: float
    ticks_per_second: float
    memory_mb: float
    max_neurons_estimate: int = 0
    max_synapses_estimate: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "neurons": self.neurons,
            "synapses": self.synapses,
            "backend": self.backend,
            "device": self.device,
            "propagate_ms": round(self.propagate_ms, 3),
            "ticks_per_second": round(self.ticks_per_second, 1),
            "memory_mb": round(self.memory_mb, 2),
            "max_neurons_estimate": self.max_neurons_estimate,
            "max_synapses_estimate": self.max_synapses_estimate,
            **self.details,
        }


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class GPUBrainEngine:
    """Sparse matrix propagation on GPU (PyTorch) or CPU (NumPy)."""

    def __init__(self, use_gpu: bool = True) -> None:
        self.backend = "cpu"
        self.device = "cpu"
        self._torch = None
        if use_gpu and _torch_available():
            import torch

            self._torch = torch
            if torch.cuda.is_available():
                self.backend = "cuda"
                self.device = torch.cuda.get_device_name(0)
            else:
                self.backend = "torch_cpu"
                self.device = "cpu"

    @property
    def gpu_available(self) -> bool:
        return self.backend == "cuda"

    def build_sparse_weights(
        self,
        n_neurons: int,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        weights: np.ndarray,
    ) -> Any:
        if self._torch is not None:
            import torch

            indices = []
            for i in range(n_neurons):
                start, end = int(row_ptr[i]), int(row_ptr[i + 1])
                for j in range(start, end):
                    indices.append([i, int(col_idx[j])])
            if not indices:
                return torch.zeros(n_neurons, n_neurons, device=self.backend if self.backend == "cuda" else "cpu")
            idx = torch.tensor(indices, dtype=torch.long).t()
            vals = torch.tensor(weights, dtype=torch.float32)
            dev = "cuda" if self.backend == "cuda" else "cpu"
            return torch.sparse_coo_tensor(idx, vals, (n_neurons, n_neurons), device=dev).coalesce()
        return (row_ptr, col_idx, weights)

    def propagate_batch(
        self,
        activation: np.ndarray,
        weights: Any,
        *,
        n_ticks: int = 10,
    ) -> tuple[np.ndarray, float]:
        """Run n_ticks propagation, return final activation and elapsed ms."""
        n = len(activation)
        t0 = time.perf_counter()

        if self._torch is not None and isinstance(weights, self._torch.Tensor):
            import torch

            dev = weights.device
            act = torch.tensor(activation, dtype=torch.float32, device=dev)
            for _ in range(n_ticks):
                act = torch.sparse.mm(weights, act.unsqueeze(1)).squeeze(1)
                act = torch.tanh(act)
                act = act * 0.85 + activation * 0.05
            result = act.cpu().numpy()
        else:
            row_ptr, col_idx, w = weights
            act = activation.copy()
            for _ in range(n_ticks):
                new_act = np.zeros(n, dtype=np.float32)
                for i in range(n):
                    start, end = int(row_ptr[i]), int(row_ptr[i + 1])
                    if start == end:
                        continue
                    new_act[i] = np.dot(w[start:end], act[col_idx[start:end]])
                act = np.tanh(new_act) * 0.85 + activation * 0.05
            result = act

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return result, elapsed_ms

    def benchmark(
        self,
        n_neurons: int,
        synapses_per_neuron: int = 50,
        *,
        n_ticks: int = 20,
        target_fps: float = 30.0,
    ) -> GPUCapacityResult:
        """Build random sparse brain and measure propagation throughput."""
        rng = np.random.default_rng(42)
        n_synapses = n_neurons * synapses_per_neuron

        rows = rng.integers(0, n_neurons, size=n_synapses, dtype=np.int32)
        cols = rng.integers(0, n_neurons, size=n_synapses, dtype=np.int32)
        weights = rng.uniform(0.01, 0.15, size=n_synapses).astype(np.float32)
        order = np.argsort(rows)
        rows, cols, weights = rows[order], cols[order], weights[order]

        row_ptr = np.zeros(n_neurons + 1, dtype=np.int32)
        for r in rows:
            row_ptr[r + 1] += 1
        row_ptr = np.cumsum(row_ptr)

        activation = rng.uniform(0.0, 0.5, size=n_neurons).astype(np.float32)
        activation[: max(1, n_neurons // 10)] = 0.8

        w = self.build_sparse_weights(n_neurons, row_ptr, cols, weights)
        _, elapsed = self.propagate_batch(activation, w, n_ticks=n_ticks)
        per_tick_ms = elapsed / n_ticks
        tps = 1000.0 / per_tick_ms if per_tick_ms > 0 else 0.0

        mem_bytes = n_neurons * 4 + n_synapses * (4 + 4 + 4)
        if self.backend == "cuda" and self._torch is not None:
            mem_bytes += n_neurons * n_neurons // synapses_per_neuron * 4

        target_ms = 1000.0 / target_fps
        scale = target_ms / per_tick_ms if per_tick_ms > 0 else 1.0
        max_neurons = int(n_neurons * scale * 0.8)
        max_synapses = max_neurons * synapses_per_neuron

        return GPUCapacityResult(
            neurons=n_neurons,
            synapses=n_synapses,
            backend=self.backend,
            device=self.device,
            propagate_ms=per_tick_ms,
            ticks_per_second=tps,
            memory_mb=mem_bytes / (1024 * 1024),
            max_neurons_estimate=max_neurons,
            max_synapses_estimate=max_synapses,
            details={"synapses_per_neuron": synapses_per_neuron, "n_ticks": n_ticks},
        )

    def estimate_capacity_for_brain(
        self,
        n_neurons: int,
        n_synapses: int,
        *,
        target_fps: float = 30.0,
    ) -> GPUCapacityResult:
        spn = max(1, n_synapses // max(1, n_neurons))
        return self.benchmark(n_neurons, synapses_per_neuron=spn, target_fps=target_fps)
