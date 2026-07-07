"""Campo neurale — il cuore fisico di CEREBRUM.

Modello: popolazione di neuroni a soglia (leaky integrate-and-fire) con
sinapsi ricorrenti plastiche (Hebbian + decadimento), modulate dalla
neurochimica (dopamina -> apprendimento, serotonina/GABA -> stabilità,
noradrenalina -> guadagno/arousal).

Backend:
- torch + CUDA se disponibile  -> gira interamente sulla GPU dell'utente
- torch CPU                    -> fallback
- numpy                        -> fallback minimale senza torch

Nessun pezzo di questo campo gira su un server remoto: tutto in locale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

_TORCH = None
_CUDA = False
try:  # torch è opzionale ma fortemente raccomandato per sfruttare la GPU
    import torch as _torch_mod

    _TORCH = _torch_mod
    _CUDA = bool(_torch_mod.cuda.is_available())
except Exception:  # pragma: no cover - dipende dall'ambiente
    _TORCH = None
    _CUDA = False

import numpy as np


def describe_backend() -> dict:
    """Descrive dove gira il cervello (per la telemetria/introspezione)."""
    if _TORCH is not None and _CUDA:
        try:
            name = _TORCH.cuda.get_device_name(0)
            total = _TORCH.cuda.get_device_properties(0).total_memory
        except Exception:  # pragma: no cover
            name, total = "cuda", 0
        return {
            "engine": "torch",
            "device": "cuda",
            "gpu": name,
            "gpu_memory_gb": round(total / (1024 ** 3), 2),
            "accelerated": True,
        }
    if _TORCH is not None:
        return {"engine": "torch", "device": "cpu", "accelerated": False}
    return {"engine": "numpy", "device": "cpu", "accelerated": False}


@dataclass
class FieldConfig:
    neurons: int = 4096
    compartments: int = 5  # dendriti per neurone (unità computazionali = neurons*compartments)
    connectivity: float = 0.02  # densità sinaptica ricorrente
    dt: float = 1.0
    tau_mem: float = 20.0  # costante di membrana (ms)
    v_threshold: float = 1.0
    v_reset: float = 0.0
    refractory: int = 2
    seed: int = 1


class NeuralField:
    """Popolazione ricorrente LIF con plasticità e neuromodulazione."""

    def __init__(self, config: Optional[FieldConfig] = None):
        self.cfg = config or FieldConfig()
        self.backend = describe_backend()
        self._use_torch = _TORCH is not None
        self._device = "cuda" if _CUDA else "cpu"
        n = self.cfg.neurons

        if self._use_torch:
            t = _TORCH
            g = t.Generator(device="cpu").manual_seed(self.cfg.seed)
            self.v = t.zeros(n, device=self._device)
            self.refrac = t.zeros(n, dtype=t.int32, device=self._device)
            self.spikes = t.zeros(n, device=self._device)
            self.trace = t.zeros(n, device=self._device)  # traccia di attività per STDP
            # matrice sinaptica sparsa densificata (piccola per neonato)
            w = t.randn(n, n, generator=g).to(self._device) * (1.0 / math.sqrt(n))
            mask = (t.rand(n, n, generator=g).to(self._device) < self.cfg.connectivity).float()
            self.w = (w * mask)
            self.w.fill_diagonal_(0.0)
        else:
            rng = np.random.default_rng(self.cfg.seed)
            self.v = np.zeros(n, dtype=np.float32)
            self.refrac = np.zeros(n, dtype=np.int32)
            self.spikes = np.zeros(n, dtype=np.float32)
            self.trace = np.zeros(n, dtype=np.float32)
            w = rng.standard_normal((n, n)).astype(np.float32) * (1.0 / math.sqrt(n))
            mask = (rng.random((n, n)) < self.cfg.connectivity).astype(np.float32)
            self.w = w * mask
            np.fill_diagonal(self.w, 0.0)

        self.step_count = 0
        self.total_spikes = 0.0

    # ---- proprietà ----
    @property
    def neurons(self) -> int:
        return self.cfg.neurons

    @property
    def computational_units(self) -> int:
        return self.cfg.neurons * self.cfg.compartments

    @property
    def synapses(self) -> int:
        if self._use_torch:
            return int((self.w != 0).sum().item())
        return int(np.count_nonzero(self.w))

    # ---- dinamica ----
    def step(self, external: Optional["np.ndarray"], neuromod: dict) -> dict:
        """Un tick di ~1ms. `external` = corrente sensoriale (len=neurons)."""
        gain = float(1.0 + 0.8 * neuromod.get("noradrenaline", 0.3))  # arousal
        lr = float(0.02 * neuromod.get("dopamine", 0.3))  # apprendimento guidato dalla ricompensa
        inhib = float(0.4 + 0.6 * neuromod.get("gaba", 0.3))  # stabilità
        # attività spontanea di fondo: il cervello non è mai a riposo assoluto.
        # ACh/noradrenalina alzano l'eccitabilità intrinseca.
        noise = float(0.35 + 0.4 * neuromod.get("acetylcholine", 0.4)
                      + 0.3 * neuromod.get("noradrenaline", 0.3))

        if self._use_torch:
            return self._step_torch(external, gain, lr, inhib, noise)
        return self._step_numpy(external, gain, lr, inhib, noise)

    def _prep_external(self, external, lib):
        n = self.cfg.neurons
        if external is None:
            if self._use_torch:
                return _TORCH.zeros(n, device=self._device)
            return np.zeros(n, dtype=np.float32)
        arr = np.asarray(external, dtype=np.float32).reshape(-1)
        if arr.shape[0] < n:
            arr = np.pad(arr, (0, n - arr.shape[0]))
        else:
            arr = arr[:n]
        if self._use_torch:
            return _TORCH.from_numpy(arr).to(self._device)
        return arr

    def _step_torch(self, external, gain, lr, inhib, noise):
        t = _TORCH
        ext = self._prep_external(external, t)
        rec = t.matmul(self.w, self.spikes) * inhib
        spontaneous = t.rand(self.cfg.neurons, device=self._device) * (0.08 * noise)
        drive = (ext * gain) + rec + spontaneous
        decay = math.exp(-self.cfg.dt / self.cfg.tau_mem)
        active = (self.refrac <= 0).float()
        self.v = (self.v * decay + drive) * active
        new_spikes = (self.v >= self.cfg.v_threshold).float()
        self.v = t.where(new_spikes > 0, t.full_like(self.v, self.cfg.v_reset), self.v)
        self.refrac = t.where(new_spikes > 0,
                              t.full_like(self.refrac, self.cfg.refractory),
                              (self.refrac - 1).clamp(min=0))
        self.trace = self.trace * 0.9 + new_spikes
        # plasticità Hebbian modulata da dopamina (pre*post) con normalizzazione
        if lr > 0:
            pre = self.trace.unsqueeze(0)
            post = new_spikes.unsqueeze(1)
            dw = (post * pre) * lr
            self.w += dw
            self.w *= 0.9999  # decadimento omeostatico
            self.w.clamp_(-4.0, 4.0)
            self.w.fill_diagonal_(0.0)
        self.spikes = new_spikes
        fired = float(new_spikes.sum().item())
        self.step_count += 1
        self.total_spikes += fired
        return {
            "fired": fired,
            "rate": fired / self.cfg.neurons,
            "mean_v": float(self.v.mean().item()),
            "activity": float(self.trace.mean().item()),
        }

    def _step_numpy(self, external, gain, lr, inhib, noise):
        ext = self._prep_external(external, np)
        rec = self.w.dot(self.spikes) * inhib
        spontaneous = np.random.random(self.cfg.neurons).astype(np.float32) * (0.08 * noise)
        drive = (ext * gain) + rec + spontaneous
        decay = math.exp(-self.cfg.dt / self.cfg.tau_mem)
        active = (self.refrac <= 0).astype(np.float32)
        self.v = (self.v * decay + drive) * active
        new_spikes = (self.v >= self.cfg.v_threshold).astype(np.float32)
        self.v = np.where(new_spikes > 0, self.cfg.v_reset, self.v)
        self.refrac = np.where(new_spikes > 0, self.cfg.refractory,
                               np.maximum(self.refrac - 1, 0))
        self.trace = self.trace * 0.9 + new_spikes
        if lr > 0:
            dw = np.outer(new_spikes, self.trace) * lr
            self.w += dw
            self.w *= 0.9999
            np.clip(self.w, -4.0, 4.0, out=self.w)
            np.fill_diagonal(self.w, 0.0)
        self.spikes = new_spikes
        fired = float(new_spikes.sum())
        self.step_count += 1
        self.total_spikes += fired
        return {
            "fired": fired,
            "rate": fired / self.cfg.neurons,
            "mean_v": float(self.v.mean()),
            "activity": float(self.trace.mean()),
        }

    def region_activity(self, regions: int = 8):
        """Attività media per 'regione' (blocchi contigui del campo)."""
        if self._use_torch:
            tr = self.trace.detach().to("cpu").numpy()
        else:
            tr = self.trace
        chunks = np.array_split(tr, regions)
        return [float(c.mean()) for c in chunks]

    def phi_estimate(self) -> float:
        """Proxy di integrazione dell'informazione (Φ): varianza tra regioni
        moderata dall'attività complessiva. Non è IIT rigoroso, è un indicatore
        operativo di quanto il campo sia insieme integrato e differenziato."""
        reg = np.asarray(self.region_activity(16), dtype=np.float64)
        if reg.sum() <= 0:
            return 0.0
        diff = float(reg.std())
        integ = float(reg.mean())
        return round(200.0 * diff * math.sqrt(max(integ, 1e-6)) * 10.0, 2)
