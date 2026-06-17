"""Kernel CUDA Numba + fallback CPU — neuroni-pixel con onde e gravità.

Texture neurone per pixel: [impulse, phase, weight, v_cache]
  - impulse: energia impulso corrente
  - phase: fase armonica locale [0, 2π)
  - weight: massa gravitazionale / memoria lungo termine
  - v_cache: velocità precedente (per accelerazione)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mindruntime.cuda_util import HAS_CUDA, HAS_NUMBA

if HAS_NUMBA:
    from numba import cuda, float32, int32
else:  # pragma: no cover
    cuda = None  # type: ignore

TWO_PI = 6.283185307179586
NEIGH_DX = (-1, 0, 1, -1, 1, -1, 0, 1)
NEIGH_DY = (-1, -1, -1, 0, 0, 1, 1, 1)
NEIGH_DIST2 = (2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0)


# --- Kernel 1: inizializzazione da frame RGB ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _initialize_neurons_cuda(rgb, neurons, threshold, seed):
        """Pixel luminoso → impulso; peso e fase pseudo-casuali coerenti per posizione."""
        y, x = cuda.grid(2)
        h, w = rgb.shape[0], rgb.shape[1]
        if y >= h or x >= w:
            return
        lum = 0.299 * rgb[y, x, 0] + 0.587 * rgb[y, x, 1] + 0.114 * rgb[y, x, 2]
        idx = y * w + x
        # LCG deterministico per riproducibilità senza RNG host
        state = (seed + idx * 1103515245 + 12345) & 0x7FFFFFFF
        r1 = (state % 10000) / 10000.0
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        r2 = (state % 10000) / 10000.0
        neurons[y, x, 1] = r1 * TWO_PI
        neurons[y, x, 2] = 0.5 + (r2 - 0.5) * 0.2  # peso ~ N(0.5, 0.1)
        neurons[y, x, 3] = 0.0
        if lum > threshold:
            neurons[y, x, 0] = min(1.0, lum)
        else:
            neurons[y, x, 0] = 0.0


# --- Kernel 2: propagazione wavefront ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _propagate_wavefront_cuda(
        prev1,
        prev2,
        out,
        decay,
        dt,
        base_freq,
        freq_scale,
        gravity_g,
        excite_thresh,
        excite_gain,
    ):
        """Onda + interferenza di fase + gravità virtuale."""
        y, x = cuda.grid(2)
        h, w = prev1.shape[0], prev1.shape[1]
        if y >= h or x >= w:
            return

        imp1 = prev1[y, x, 0]
        imp2 = prev2[y, x, 0]
        phase = prev1[y, x, 1]
        weight = prev1[y, x, 2]
        v_prev = prev1[y, x, 3]

        v = imp1 - imp2
        a = v - v_prev

        wave_sum = 0.0
        gravity_pull = 0.0
        for k in range(8):
            nx = x + NEIGH_DX[k]
            ny = y + NEIGH_DY[k]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            n_imp = prev1[ny, nx, 0]
            n_phase = prev1[ny, nx, 1]
            n_weight = prev1[ny, nx, 2]
            dphase = n_phase - phase
            wave_sum += n_imp * math.cos(dphase)
            if n_weight > weight:
                gravity_pull += (n_weight - weight) / NEIGH_DIST2[k]

        impulse = imp1 * decay + wave_sum * 0.12 * dt + a * 0.35 * dt
        impulse += gravity_pull * gravity_g * dt
        if wave_sum > excite_thresh:
            impulse += excite_gain * (wave_sum - excite_thresh)

        new_phase = phase + base_freq + impulse * freq_scale
        new_phase = new_phase - TWO_PI * math.floor(new_phase / TWO_PI)

        out[y, x, 0] = min(1.0, max(0.0, impulse))
        out[y, x, 1] = new_phase
        out[y, x, 2] = weight
        out[y, x, 3] = v


# --- Kernel 3: plasticità Hebbiana sui pesi ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _update_weights_hebbian_cuda(state, out_weights, hebb_rate, max_weight):
        """Correlazione temporale locale → aumento peso (attrattore)."""
        y, x = cuda.grid(2)
        h, w = state.shape[0], state.shape[1]
        if y >= h or x >= w:
            return
        imp = state[y, x, 0]
        w = state[y, x, 2]
        corr = 0.0
        for k in range(8):
            nx = x + NEIGH_DX[k]
            ny = y + NEIGH_DY[k]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            n_imp = state[ny, nx, 0]
            corr += imp * n_imp
        if corr > 0.02:
            w = min(max_weight, w + hebb_rate * corr)
        out_weights[y, x] = w


# --- Kernel 4: matching risonatori (dominio spaziale, normalizzato) ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _match_resonators_cuda(impulse_map, templates, scores, tpl_h, tpl_w):
        """Correlazione normalizzata impulsi vs template (N template)."""
        tid = cuda.grid(1)
        n_tpl = templates.shape[0]
        if tid >= n_tpl:
            return
        h, w = impulse_map.shape[0], impulse_map.shape[1]
        y0 = (h - tpl_h) // 2
        x0 = (w - tpl_w) // 2
        dot = 0.0
        na = 0.0
        nb = 0.0
        for ty in range(tpl_h):
            for tx in range(tpl_w):
                a = impulse_map[y0 + ty, x0 + tx]
                b = templates[tid, ty, tx]
                dot += a * b
                na += a * a
                nb += b * b
        denom = math.sqrt(na * nb) + 1e-9
        scores[tid] = dot / denom


# --- API host ---

def initialize_neurons(
    rgb: np.ndarray,
    neurons: np.ndarray,
    *,
    threshold: float = 0.18,
    seed: int = 42,
) -> None:
    """Inizializza texture neuroni da frame RGB uint8/float."""
    rgb = np.ascontiguousarray(rgb.astype(np.float32))
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    h, w = rgb.shape[:2]
    assert neurons.shape == (h, w, 4)
    if HAS_CUDA and cuda is not None:
        threads = (16, 16)
        blocks = ((w + 15) // 16, (h + 15) // 16)
        _initialize_neurons_cuda[blocks, threads](rgb, neurons, float32(threshold), int32(seed))
        cuda.synchronize()
    else:
        _initialize_neurons_cpu(rgb, neurons, threshold=threshold, seed=seed)


def propagate_wavefront(
    prev1: np.ndarray,
    prev2: np.ndarray,
    out: np.ndarray,
    *,
    decay: float = 0.95,
    dt: float = 1.0,
    base_freq: float = 0.08,
    freq_scale: float = 0.35,
    gravity_g: float = 0.06,
    excite_thresh: float = 0.25,
    excite_gain: float = 0.18,
) -> None:
    """Un passo di propagazione onda con triple-buffer."""
    if HAS_CUDA and cuda is not None:
        h, w = prev1.shape[:2]
        threads = (16, 16)
        blocks = ((w + 15) // 16, (h + 15) // 16)
        _propagate_wavefront_cuda[blocks, threads](
            prev1,
            prev2,
            out,
            float32(decay),
            float32(dt),
            float32(base_freq),
            float32(freq_scale),
            float32(gravity_g),
            float32(excite_thresh),
            float32(excite_gain),
        )
        cuda.synchronize()
    else:
        _propagate_wavefront_cpu(
            prev1,
            prev2,
            out,
            decay=decay,
            dt=dt,
            base_freq=base_freq,
            freq_scale=freq_scale,
            gravity_g=gravity_g,
            excite_thresh=excite_thresh,
            excite_gain=excite_gain,
        )


def update_weights_hebbian(
    state: np.ndarray,
    out_weights: np.ndarray | None = None,
    *,
    hebb_rate: float = 0.012,
    max_weight: float = 2.5,
) -> np.ndarray:
    """Aggiorna pesi gravitazionali; ritorna array pesi."""
    h, w = state.shape[:2]
    weights = state[:, :, 2].copy() if out_weights is None else out_weights
    if HAS_CUDA and cuda is not None:
        _update_weights_hebbian_cuda[((w + 15) // 16, (h + 15) // 16), (16, 16)](
            state, weights, float32(hebb_rate), float32(max_weight)
        )
        cuda.synchronize()
        state[:, :, 2] = weights
    else:
        _update_weights_hebbian_cpu(state, weights, hebb_rate=hebb_rate, max_weight=max_weight)
        state[:, :, 2] = weights
    return weights


def match_resonators(
    impulse_map: np.ndarray,
    templates: np.ndarray,
) -> np.ndarray:
    """Ritorna score [N] per N template."""
    impulse_map = np.ascontiguousarray(impulse_map.astype(np.float32))
    templates = np.ascontiguousarray(templates.astype(np.float32))
    n_tpl, tpl_h, tpl_w = templates.shape
    scores = np.zeros(n_tpl, dtype=np.float32)
    if HAS_CUDA and cuda is not None:
        _match_resonators_cuda[(n_tpl + 255) // 256, 256](
            impulse_map, templates, scores, int32(tpl_h), int32(tpl_w)
        )
        cuda.synchronize()
    else:
        _match_resonators_cpu(impulse_map, templates, scores)
    return scores


# --- CPU fallback (identica logica, per test e macchine senza CUDA) ---

def _lcg(seed: int, idx: int) -> tuple[float, float]:
    state = (seed + idx * 1103515245 + 12345) & 0x7FFFFFFF
    r1 = (state % 10000) / 10000.0
    state = (state * 1103515245 + 12345) & 0x7FFFFFFF
    r2 = (state % 10000) / 10000.0
    return r1, r2


def _initialize_neurons_cpu(
    rgb: np.ndarray,
    neurons: np.ndarray,
    *,
    threshold: float,
    seed: int,
) -> None:
    h, w = rgb.shape[:2]
    for y in range(h):
        for x in range(w):
            lum = 0.299 * rgb[y, x, 0] + 0.587 * rgb[y, x, 1] + 0.114 * rgb[y, x, 2]
            r1, r2 = _lcg(seed, y * w + x)
            neurons[y, x, 1] = r1 * TWO_PI
            neurons[y, x, 2] = 0.5 + (r2 - 0.5) * 0.2
            neurons[y, x, 3] = 0.0
            neurons[y, x, 0] = min(1.0, lum) if lum > threshold else 0.0


def _propagate_wavefront_cpu(
    prev1: np.ndarray,
    prev2: np.ndarray,
    out: np.ndarray,
    **kw: Any,
) -> None:
    decay = kw["decay"]
    dt = kw["dt"]
    base_freq = kw["base_freq"]
    freq_scale = kw["freq_scale"]
    gravity_g = kw["gravity_g"]
    excite_thresh = kw["excite_thresh"]
    excite_gain = kw["excite_gain"]
    h, w = prev1.shape[:2]
    for y in range(h):
        for x in range(w):
            imp1 = prev1[y, x, 0]
            imp2 = prev2[y, x, 0]
            phase = prev1[y, x, 1]
            weight = prev1[y, x, 2]
            v_prev = prev1[y, x, 3]
            v = imp1 - imp2
            a = v - v_prev
            wave_sum = 0.0
            gravity_pull = 0.0
            for k in range(8):
                nx, ny = x + NEIGH_DX[k], y + NEIGH_DY[k]
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                n_imp = prev1[ny, nx, 0]
                n_phase = prev1[ny, nx, 1]
                n_weight = prev1[ny, nx, 2]
                wave_sum += n_imp * math.cos(n_phase - phase)
                if n_weight > weight:
                    gravity_pull += (n_weight - weight) / NEIGH_DIST2[k]
            impulse = imp1 * decay + wave_sum * 0.12 * dt + a * 0.35 * dt
            impulse += gravity_pull * gravity_g * dt
            if wave_sum > excite_thresh:
                impulse += excite_gain * (wave_sum - excite_thresh)
            new_phase = phase + base_freq + impulse * freq_scale
            new_phase = new_phase - TWO_PI * math.floor(new_phase / TWO_PI)
            out[y, x, 0] = min(1.0, max(0.0, impulse))
            out[y, x, 1] = new_phase
            out[y, x, 2] = weight
            out[y, x, 3] = v


def _update_weights_hebbian_cpu(
    state: np.ndarray,
    weights: np.ndarray,
    *,
    hebb_rate: float,
    max_weight: float,
) -> None:
    h, w = state.shape[:2]
    for y in range(h):
        for x in range(w):
            imp = state[y, x, 0]
            wv = state[y, x, 2]
            corr = 0.0
            for k in range(8):
                nx, ny = x + NEIGH_DX[k], y + NEIGH_DY[k]
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                corr += imp * state[ny, nx, 0]
            if corr > 0.02:
                wv = min(max_weight, wv + hebb_rate * corr)
            weights[y, x] = wv


def _match_resonators_cpu(
    impulse_map: np.ndarray,
    templates: np.ndarray,
    scores: np.ndarray,
) -> None:
    n_tpl, tpl_h, tpl_w = templates.shape
    h, w = impulse_map.shape
    y0, x0 = (h - tpl_h) // 2, (w - tpl_w) // 2
    patch = impulse_map[y0 : y0 + tpl_h, x0 : x0 + tpl_w].ravel()
    na = float(np.dot(patch, patch)) + 1e-9
    for i in range(n_tpl):
        t = templates[i].ravel()
        nb = float(np.dot(t, t)) + 1e-9
        scores[i] = float(np.dot(patch, t) / math.sqrt(na * nb))
