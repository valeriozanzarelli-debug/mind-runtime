"""Campo neurale strutturato — il cuore fisico di CEREBRUM (v2).

Rispetto alla v1 (matrice densa uniforme), questo substrato adotta i principi
di un cervello vero:

1. CONNETTOMA STRUTTURATO: neuroni divisi in regioni con ruoli distinti
   (talamo = hub/attenzione, corteccia sensoriale, corteccia associativa,
   ippocampo = memoria, prefrontale = memoria di lavoro/top-down) e cablaggio
   specifico regione->regione, non una zuppa uniforme.
2. SINAPSI SPARSE + EVENT-DRIVEN: le sinapsi sono liste (formato CSR), e ad
   ogni tick propagano SOLO i neuroni che hanno sparato -> costo ~O(spike*fanout)
   invece di O(N^2). E' cosi' che un cervello resta efficiente.
3. E/I BALANCE (legge di Dale): ~20% di neuroni inibitori (peso in uscita
   negativo) per stabilizzare l'attivita'.
4. PLASTICITA' A TRE FATTORI: tracce di eleggibilita' per sinapsi (pre*post)
   che diventano cambiamento di peso solo quando arriva il terzo fattore, la
   dopamina (ricompensa/sorpresa). E' apprendimento locale, online, senza
   backprop globale.

Backend: torch+CUDA (GPU dell'utente) se disponibile, altrimenti numpy.
Nessun pezzo gira su server remoto.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple

_TORCH = None
_CUDA = False
try:
    import torch as _torch_mod

    _TORCH = _torch_mod
    _CUDA = bool(_torch_mod.cuda.is_available())
except Exception:  # pragma: no cover
    _TORCH = None
    _CUDA = False

import numpy as np


def describe_backend() -> dict:
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


# Layout delle regioni (frazione dei neuroni) e ruolo.
REGION_LAYOUT: List[Tuple[str, float]] = [
    ("sensory", 0.20),      # corteccia sensoriale: riceve input esterno
    ("thalamus", 0.08),     # hub: instrada + guadagno attenzionale
    ("association", 0.42),  # corteccia associativa: ricorrente, "pensiero"
    ("hippocampus", 0.16),  # memoria episodica / recall
    ("prefrontal", 0.14),   # memoria di lavoro / controllo top-down
]

# Cablaggio regione->regione: (prob_connessione, scala_peso).
REGION_WIRING: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("sensory", "thalamus"): (0.10, 1.0),
    ("thalamus", "sensory"): (0.04, 0.6),      # feedback attenzionale
    ("thalamus", "association"): (0.06, 1.0),
    ("association", "association"): (0.03, 0.8),  # ricorrenza
    ("association", "hippocampus"): (0.05, 1.0),
    ("hippocampus", "association"): (0.05, 0.9),  # recall
    ("association", "prefrontal"): (0.05, 0.9),
    ("prefrontal", "association"): (0.05, 0.7),   # top-down / predizione
    ("prefrontal", "thalamus"): (0.03, 0.6),      # controllo attenzione
}


@dataclass
class FieldConfig:
    neurons: int = 4096
    compartments: int = 5
    dt: float = 1.0
    tau_mem: float = 20.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    refractory: int = 2
    inhib_fraction: float = 0.2
    # guadagno dei neuroni inibitori: >1 perche' un cervello e' tenuto stabile
    # da inibizione che domina leggermente (previene l'attivita' a valanga).
    inhib_gain: float = 6.0
    trace_decay: float = 0.9
    elig_decay: float = 0.95
    # eccitabilita' spontanea a riposo: mantiene il cervello attivo anche
    # senza stimoli (attivita' di fondo corticale). Calibrata per un tasso di
    # scarica a riposo di pochi punti percentuali.
    rest_drive: float = 0.085
    rest_jitter: float = 0.06
    seed: int = 1


class NeuralField:
    def __init__(self, config: Optional[FieldConfig] = None):
        self.cfg = config or FieldConfig()
        self.backend = describe_backend()
        self._use_torch = _TORCH is not None
        self._device = "cuda" if _CUDA else "cpu"
        self.dopamine_baseline = 0.3

        self._build_regions()
        self._build_synapses()
        self._init_state()

        self.step_count = 0
        self.total_spikes = 0.0

    # ---------- costruzione ----------
    def _build_regions(self) -> None:
        n = self.cfg.neurons
        self.regions: Dict[str, Tuple[int, int]] = {}
        start = 0
        for i, (name, frac) in enumerate(REGION_LAYOUT):
            size = max(1, int(round(n * frac)))
            if i == len(REGION_LAYOUT) - 1:
                size = n - start  # l'ultima prende il resto
            self.regions[name] = (start, start + size)
            start += size
        # segno dei neuroni (legge di Dale): alcuni inibitori
        rng = np.random.default_rng(self.cfg.seed)
        self.neuron_sign = np.ones(n, dtype=np.float32)
        inh = rng.random(n) < self.cfg.inhib_fraction
        self.neuron_sign[inh] = -self.cfg.inhib_gain

    def _build_synapses(self) -> None:
        rng = np.random.default_rng(self.cfg.seed + 1)
        pre_list, post_list, w_list = [], [], []
        for (src, dst), (prob, scale) in REGION_WIRING.items():
            s0, s1 = self.regions[src]
            d0, d1 = self.regions[dst]
            src_ids = np.arange(s0, s1)
            dst_ids = np.arange(d0, d1)
            # per ogni neurone sorgente, campiona un fan-out verso la destinazione
            fan = max(1, int(len(dst_ids) * prob))
            for pre in src_ids:
                targets = rng.choice(dst_ids, size=min(fan, len(dst_ids)), replace=False)
                targets = targets[targets != pre]
                if len(targets) == 0:
                    continue
                weights = np.abs(rng.normal(0.0, scale / math.sqrt(fan + 1), len(targets)))
                pre_list.append(np.full(len(targets), pre, dtype=np.int64))
                post_list.append(targets.astype(np.int64))
                w_list.append(weights.astype(np.float32))

        self._pre_np = np.concatenate(pre_list) if pre_list else np.zeros(0, np.int64)
        self._post_np = np.concatenate(post_list) if post_list else np.zeros(0, np.int64)
        self._w_np = np.concatenate(w_list) if w_list else np.zeros(0, np.float32)
        self.w_max = 4.0

        if self._use_torch:
            t = _TORCH
            self.pre_idx = t.from_numpy(self._pre_np).to(self._device)
            self.post_idx = t.from_numpy(self._post_np).to(self._device)
            self.w = t.from_numpy(self._w_np).to(self._device)
            self.sign = t.from_numpy(self.neuron_sign).to(self._device)
            self.elig = t.zeros_like(self.w)
        else:
            self.pre_idx = self._pre_np
            self.post_idx = self._post_np
            self.w = self._w_np.copy()
            self.sign = self.neuron_sign
            self.elig = np.zeros_like(self.w)

    def _init_state(self) -> None:
        n = self.cfg.neurons
        if self._use_torch:
            t = _TORCH
            z = lambda: t.zeros(n, device=self._device)
            self.v = z()
            self.spikes = z()
            self.refrac = t.zeros(n, dtype=t.int32, device=self._device)
            self.pre_trace = z()
            self.post_trace = z()
            self.trace = z()
        else:
            self.v = np.zeros(n, dtype=np.float32)
            self.spikes = np.zeros(n, dtype=np.float32)
            self.refrac = np.zeros(n, dtype=np.int32)
            self.pre_trace = np.zeros(n, dtype=np.float32)
            self.post_trace = np.zeros(n, dtype=np.float32)
            self.trace = np.zeros(n, dtype=np.float32)

    # ---------- proprieta' ----------
    @property
    def neurons(self) -> int:
        return self.cfg.neurons

    @property
    def computational_units(self) -> int:
        return self.cfg.neurons * self.cfg.compartments

    @property
    def synapses(self) -> int:
        return int(self._w_np.shape[0])

    def _sensory_external(self, external) -> "np.ndarray":
        """Mappa un vettore sensoriale (qualsiasi lunghezza) sulla regione
        sensoriale, come corrente in ingresso."""
        s0, s1 = self.regions["sensory"]
        size = s1 - s0
        out = np.zeros(self.cfg.neurons, dtype=np.float32)
        if external is None:
            return out
        arr = np.asarray(external, dtype=np.float32).reshape(-1)
        if arr.shape[0] == 0:
            return out
        # ridimensiona (tile/troncamento) alla dimensione della regione
        if arr.shape[0] < size:
            reps = int(np.ceil(size / arr.shape[0]))
            arr = np.tile(arr, reps)[:size]
        else:
            arr = arr[:size]
        out[s0:s1] = arr
        return out

    # ---------- dinamica ----------
    def step(self, external, neuromod: dict) -> dict:
        gain = float(1.0 + 0.8 * neuromod.get("noradrenaline", 0.3))
        lr = float(0.05)
        dopamine = float(neuromod.get("dopamine", 0.3))
        ach = float(neuromod.get("acetylcholine", 0.4))
        na = float(neuromod.get("noradrenaline", 0.3))
        # drive di fondo: base costante (resting) + jitter modulato da ACh/NA
        # (attenzione/arousal alzano l'eccitabilita', come nel cervello vero).
        rest = self.cfg.rest_drive + self.cfg.rest_jitter * (0.5 * ach + 0.5 * na)
        ext = self._sensory_external(external) * gain
        if self._use_torch:
            return self._step_torch(ext, rest, lr, dopamine)
        return self._step_numpy(ext, rest, lr, dopamine)

    def _step_numpy(self, ext, rest, lr, dopamine):
        n = self.cfg.neurons
        # propagazione event-driven: solo i neuroni che hanno sparato contribuiscono
        current = np.zeros(n, dtype=np.float32)
        if self.spikes.any() and self._pre_np.shape[0] > 0:
            active_syn = self.spikes[self._pre_np] > 0
            if active_syn.any():
                contrib = (self.sign[self._pre_np][active_syn]
                           * self.w[active_syn])
                np.add.at(current, self._post_np[active_syn], contrib)
        spontaneous = np.random.random(n).astype(np.float32) * rest
        drive = ext + current + spontaneous

        decay = math.exp(-self.cfg.dt / self.cfg.tau_mem)
        active = (self.refrac <= 0).astype(np.float32)
        self.v = (self.v * decay + drive) * active
        new_spikes = (self.v >= self.cfg.v_threshold).astype(np.float32)
        self.v = np.where(new_spikes > 0, self.cfg.v_reset, self.v)
        self.refrac = np.where(new_spikes > 0, self.cfg.refractory,
                               np.maximum(self.refrac - 1, 0))

        # tracce
        td = self.cfg.trace_decay
        self.pre_trace = self.pre_trace * td + new_spikes
        self.post_trace = self.post_trace * td + new_spikes
        self.trace = self.trace * td + new_spikes

        self._plasticity_numpy(new_spikes, lr, dopamine)
        self.spikes = new_spikes

        fired = float(new_spikes.sum())
        self.step_count += 1
        self.total_spikes += fired
        return {"fired": fired, "rate": fired / n,
                "mean_v": float(self.v.mean()), "activity": float(self.trace.mean())}

    def _plasticity_numpy(self, new_spikes, lr, dopamine):
        if self._pre_np.shape[0] == 0:
            return
        pre = self._pre_np
        post = self._post_np
        # STDP: LTP se pre precede post (pre_trace alto * post spara);
        #       LTD se post precede pre (post_trace alto * pre spara).
        ltp = self.pre_trace[pre] * new_spikes[post]
        ltd = self.post_trace[post] * new_spikes[pre]
        self.elig = self.elig * self.cfg.elig_decay + (ltp - ltd)
        # terzo fattore: la dopamina relativa al baseline "consolida" l'eleggibilita'
        modulator = dopamine - self.dopamine_baseline
        self.w += lr * modulator * self.elig
        np.clip(self.w, 0.0, self.w_max, out=self.w)

    def _step_torch(self, ext, rest, lr, dopamine):
        t = _TORCH
        n = self.cfg.neurons
        ext_t = t.from_numpy(ext).to(self._device)
        current = t.zeros(n, device=self._device)
        if self._pre_np.shape[0] > 0 and float(self.spikes.sum()) > 0:
            active = self.spikes[self.pre_idx] > 0
            if bool(active.any()):
                contrib = self.sign[self.pre_idx][active] * self.w[active]
                current.index_add_(0, self.post_idx[active], contrib)
        spontaneous = t.rand(n, device=self._device) * rest
        drive = ext_t + current + spontaneous

        decay = math.exp(-self.cfg.dt / self.cfg.tau_mem)
        active_mask = (self.refrac <= 0).float()
        self.v = (self.v * decay + drive) * active_mask
        new_spikes = (self.v >= self.cfg.v_threshold).float()
        self.v = t.where(new_spikes > 0, t.full_like(self.v, self.cfg.v_reset), self.v)
        self.refrac = t.where(new_spikes > 0,
                              t.full_like(self.refrac, self.cfg.refractory),
                              (self.refrac - 1).clamp(min=0))
        td = self.cfg.trace_decay
        self.pre_trace = self.pre_trace * td + new_spikes
        self.post_trace = self.post_trace * td + new_spikes
        self.trace = self.trace * td + new_spikes

        if self._pre_np.shape[0] > 0:
            ltp = self.pre_trace[self.pre_idx] * new_spikes[self.post_idx]
            ltd = self.post_trace[self.post_idx] * new_spikes[self.pre_idx]
            self.elig = self.elig * self.cfg.elig_decay + (ltp - ltd)
            modulator = dopamine - self.dopamine_baseline
            self.w = (self.w + lr * modulator * self.elig).clamp(0.0, self.w_max)

        self.spikes = new_spikes
        fired = float(new_spikes.sum().item())
        self.step_count += 1
        self.total_spikes += fired
        return {"fired": fired, "rate": fired / n,
                "mean_v": float(self.v.mean().item()),
                "activity": float(self.trace.mean().item())}

    # ---------- letture ----------
    def _trace_np(self):
        if self._use_torch:
            return self.trace.detach().to("cpu").numpy()
        return self.trace

    def region_activity(self, regions: int = 8):
        tr = self._trace_np()
        return [float(c.mean()) for c in np.array_split(tr, regions)]

    def region_activity_named(self) -> Dict[str, float]:
        tr = self._trace_np()
        out = {}
        for name, (a, b) in self.regions.items():
            seg = tr[a:b]
            out[name] = float(seg.mean()) if len(seg) else 0.0
        return out

    def assoc_activity(self, dim: int = 64):
        """Vettore compatto dell'attivita' associativa (input al predittore)."""
        a, b = self.regions["association"]
        tr = self._trace_np()[a:b]
        if len(tr) == 0:
            return np.zeros(dim, dtype=np.float32)
        return np.array([float(c.mean()) for c in np.array_split(tr, dim)], dtype=np.float32)

    def phi_estimate(self) -> float:
        """Proxy operativo di integrazione dell'informazione (Φ).

        Un cervello cosciente e' insieme INTEGRATO (le regioni cooperano, alta
        attivita' media) e DIFFERENZIATO (le regioni non fanno tutte la stessa
        cosa, varieta' tra regioni). Combiniamo i due, normalizzati, su una
        scala ~0..200 (0 = spento/uniforme, ~100-175 = ricco e integrato)."""
        reg = np.asarray(self.region_activity(16), dtype=np.float64)
        m = float(reg.mean())
        if m <= 1e-6:
            return 0.0
        integration = math.tanh(m * 12.0)          # 0..1: quanto e' attivo
        cv = float(reg.std()) / (m + 1e-6)          # coeff. variazione
        differentiation = math.tanh(cv * 1.5)       # 0..1: quanto e' vario
        return round(200.0 * integration * differentiation, 2)
