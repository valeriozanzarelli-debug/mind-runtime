"""Compartimenti dendritici — canali ionici virtuali + backward loop.

Canali per pixel (8):
  0 impulse   1 phase   2 weight   3 v_cache
  4 na        5 k       6 ca       7 backward
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mindruntime.cuda_util import HAS_CUDA, HAS_NUMBA
from mindruntime.gpu_core import NEIGH_DIST2, NEIGH_DX, NEIGH_DY, TWO_PI, match_resonators

if HAS_NUMBA:
    from numba import cuda, float32, int32
else:  # pragma: no cover
    cuda = None  # type: ignore

N_CHANNELS = 8
CH_IMP, CH_PH, CH_W, CH_V = 0, 1, 2, 3
CH_NA, CH_K, CH_CA, CH_BW = 4, 5, 6, 7


def compartment_shape(h: int, w: int) -> tuple[int, int, int]:
    return (h, w, N_CHANNELS)


if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _init_dendrites_cuda(rgb, state, threshold, seed):
        y, x = cuda.grid(2)
        h, w = rgb.shape[0], rgb.shape[1]
        if y >= h or x >= w:
            return
        lum = 0.299 * rgb[y, x, 0] + 0.587 * rgb[y, x, 1] + 0.114 * rgb[y, x, 2]
        idx = y * w + x
        st = (seed + idx * 1103515245 + 12345) & 0x7FFFFFFF
        r1 = (st % 10000) / 10000.0
        st = (st * 1103515245 + 12345) & 0x7FFFFFFF
        r2 = (st % 10000) / 10000.0
        state[y, x, CH_PH] = r1 * TWO_PI
        state[y, x, CH_W] = 0.45 + (r2 - 0.5) * 0.15
        state[y, x, CH_V] = 0.0
        state[y, x, CH_NA] = 0.08
        state[y, x, CH_K] = 0.55
        state[y, x, CH_CA] = 0.05
        state[y, x, CH_BW] = 0.0
        state[y, x, CH_IMP] = min(1.0, lum) if lum > threshold else 0.0

    @cuda.jit
    def _forward_dendrite_cuda(prev1, prev2, out, dt, decay, gravity_g, phase_thresh):
        """Forward: wavefront + Hodgkin-Huxley semplificato + gating da coerenza fase."""
        y, x = cuda.grid(2)
        h, w = prev1.shape[0], prev1.shape[1]
        if y >= h or x >= w:
            return

        imp1 = prev1[y, x, CH_IMP]
        imp2 = prev2[y, x, CH_IMP]
        phase = prev1[y, x, CH_PH]
        weight = prev1[y, x, CH_W]
        v_prev = prev1[y, x, CH_V]
        na = prev1[y, x, CH_NA]
        k = prev1[y, x, CH_K]
        ca = prev1[y, x, CH_CA]

        v = imp1 - imp2
        a = v - v_prev

        wave_sum = 0.0
        coh_sum = 0.0
        coh_n = 0.0
        gravity_pull = 0.0
        for ki in range(8):
            nx = x + NEIGH_DX[ki]
            ny = y + NEIGH_DY[ki]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            n_imp = prev1[ny, nx, CH_IMP]
            n_ph = prev1[ny, nx, CH_PH]
            n_w = prev1[ny, nx, CH_W]
            dph = n_ph - phase
            c = math.cos(dph)
            wave_sum += n_imp * c
            coh_sum += c
            coh_n += 1.0
            if n_w > weight:
                gravity_pull += (n_w - weight) / NEIGH_DIST2[ki]

        coherence = coh_sum / coh_n if coh_n > 0 else 0.0
        gate = 1.0 if coherence > phase_thresh else 0.35

        drive = imp1 + wave_sum * 0.14 * gate + a * 0.3
        m_na = 1.0 / (1.0 + math.exp(-5.0 * (drive - 0.28)))
        na = na + (m_na - na) * 0.16
        k_inf = 0.25 + 0.55 * na
        k = k + (k_inf - k) * 0.12
        ca_inf = max(0.0, coherence) * (0.15 + imp1 * 0.5)
        ca = ca + (ca_inf - ca) * 0.18

        impulse = na * (1.0 - 0.72 * k) * (1.0 + 0.45 * ca)
        impulse = impulse * decay + wave_sum * 0.1 * gate * dt
        impulse += gravity_pull * gravity_g * dt + a * 0.28 * dt
        impulse = min(1.0, max(0.0, impulse))

        new_ph = phase + 0.09 + impulse * 0.32 + ca * 0.15
        new_ph = new_ph - TWO_PI * math.floor(new_ph / TWO_PI)

        out[y, x, CH_IMP] = impulse
        out[y, x, CH_PH] = new_ph
        out[y, x, CH_W] = weight
        out[y, x, CH_V] = v
        out[y, x, CH_NA] = min(1.0, na)
        out[y, x, CH_K] = min(1.0, k)
        out[y, x, CH_CA] = min(1.0, ca)
        out[y, x, CH_BW] = prev1[y, x, CH_BW] * 0.92

    @cuda.jit
    def _backward_dendrite_cuda(state, bw_rate, max_weight):
        """Backward: feedback dendritico → rinforzo peso nelle zone coerenti."""
        y, x = cuda.grid(2)
        h, w = state.shape[0], state.shape[1]
        if y >= h or x >= w:
            return

        imp = state[y, x, CH_IMP]
        phase = state[y, x, CH_PH]
        ca = state[y, x, CH_CA]
        w = state[y, x, CH_W]

        feedback = 0.0
        for ki in range(8):
            nx = x + NEIGH_DX[ki]
            ny = y + NEIGH_DY[ki]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            n_imp = state[ny, nx, CH_IMP]
            n_ph = state[ny, nx, CH_PH]
            c = math.cos(n_ph - phase)
            if c > 0.55:
                feedback += n_imp * c * ca

        bw = feedback * 0.25
        state[y, x, CH_BW] = bw
        if bw > 0.03:
            w = min(max_weight, w + bw_rate * bw)
        state[y, x, CH_W] = w


def initialize_dendrites(
    rgb: np.ndarray,
    state: np.ndarray,
    *,
    threshold: float = 0.18,
    seed: int = 42,
) -> None:
    rgb = np.ascontiguousarray(rgb.astype(np.float32))
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    h, w = rgb.shape[:2]
    if HAS_CUDA and cuda is not None:
        _init_dendrites_cuda[((w + 15) // 16, (h + 15) // 16), (16, 16)](
            rgb, state, float32(threshold), int32(seed)
        )
        cuda.synchronize()
    else:
        _init_dendrites_cpu(rgb, state, threshold=threshold, seed=seed)


def forward_dendrite(
    prev1: np.ndarray,
    prev2: np.ndarray,
    out: np.ndarray,
    *,
    dt: float = 1.0,
    decay: float = 0.94,
    gravity_g: float = 0.07,
    phase_thresh: float = 0.45,
) -> None:
    if HAS_CUDA and cuda is not None:
        h, w = prev1.shape[:2]
        _forward_dendrite_cuda[((w + 15) // 16, (h + 15) // 16), (16, 16)](
            prev1,
            prev2,
            out,
            float32(dt),
            float32(decay),
            float32(gravity_g),
            float32(phase_thresh),
        )
        cuda.synchronize()
    else:
        _forward_dendrite_cpu(prev1, prev2, out, dt=dt, decay=decay, gravity_g=gravity_g, phase_thresh=phase_thresh)


def backward_dendrite(
    state: np.ndarray,
    *,
    bw_rate: float = 0.014,
    max_weight: float = 2.8,
) -> None:
    if HAS_CUDA and cuda is not None:
        h, w = state.shape[:2]
        _backward_dendrite_cuda[((w + 15) // 16, (h + 15) // 16), (16, 16)](
            state, float32(bw_rate), float32(max_weight)
        )
        cuda.synchronize()
    else:
        _backward_dendrite_cpu(state, bw_rate=bw_rate, max_weight=max_weight)


def coherence_map(state: np.ndarray) -> np.ndarray:
    """Mappa coerenza di fase locale (zone attive / attenzione)."""
    h, w = state.shape[:2]
    phase = state[:, :, CH_PH]
    imp = state[:, :, CH_IMP]
    out = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            ph = phase[y, x]
            s = 0.0
            n = 0
            for ki in range(8):
                nx, ny = x + NEIGH_DX[ki], y + NEIGH_DY[ki]
                if 0 <= nx < w and 0 <= ny < h:
                    s += math.cos(phase[ny, nx] - ph)
                    n += 1
            out[y, x] = (s / max(1, n)) * imp[y, x]
    return out


# re-export for engine
__all__ = [
    "N_CHANNELS",
    "CH_IMP",
    "CH_PH",
    "CH_W",
    "CH_V",
    "CH_NA",
    "CH_K",
    "CH_CA",
    "CH_BW",
    "initialize_dendrites",
    "forward_dendrite",
    "backward_dendrite",
    "coherence_map",
    "match_resonators",
    "compartment_shape",
]


def _lcg(seed: int, idx: int) -> tuple[float, float]:
    st = (seed + idx * 1103515245 + 12345) & 0x7FFFFFFF
    r1 = (st % 10000) / 10000.0
    st = (st * 1103515245 + 12345) & 0x7FFFFFFF
    r2 = (st % 10000) / 10000.0
    return r1, r2


def _init_dendrites_cpu(rgb, state, *, threshold, seed):
    h, w = rgb.shape[:2]
    for y in range(h):
        for x in range(w):
            lum = 0.299 * rgb[y, x, 0] + 0.587 * rgb[y, x, 1] + 0.114 * rgb[y, x, 2]
            r1, r2 = _lcg(seed, y * w + x)
            state[y, x, CH_PH] = r1 * TWO_PI
            state[y, x, CH_W] = 0.45 + (r2 - 0.5) * 0.15
            state[y, x, CH_V] = 0.0
            state[y, x, CH_NA] = 0.08
            state[y, x, CH_K] = 0.55
            state[y, x, CH_CA] = 0.05
            state[y, x, CH_BW] = 0.0
            state[y, x, CH_IMP] = min(1.0, lum) if lum > threshold else 0.0


def _forward_dendrite_cpu(prev1, prev2, out, **kw: Any):
    dt = kw["dt"]
    decay = kw["decay"]
    gravity_g = kw["gravity_g"]
    phase_thresh = kw["phase_thresh"]
    h, w = prev1.shape[:2]
    for y in range(h):
        for x in range(w):
            imp1 = prev1[y, x, CH_IMP]
            imp2 = prev2[y, x, CH_IMP]
            phase = prev1[y, x, CH_PH]
            weight = prev1[y, x, CH_W]
            v_prev = prev1[y, x, CH_V]
            na = prev1[y, x, CH_NA]
            k = prev1[y, x, CH_K]
            ca = prev1[y, x, CH_CA]
            v = imp1 - imp2
            a = v - v_prev
            wave_sum = 0.0
            coh_sum = 0.0
            coh_n = 0.0
            gravity_pull = 0.0
            for ki in range(8):
                nx, ny = x + NEIGH_DX[ki], y + NEIGH_DY[ki]
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                n_imp = prev1[ny, nx, CH_IMP]
                n_ph = prev1[ny, nx, CH_PH]
                n_w = prev1[ny, nx, CH_W]
                c = math.cos(n_ph - phase)
                wave_sum += n_imp * c
                coh_sum += c
                coh_n += 1.0
                if n_w > weight:
                    gravity_pull += (n_w - weight) / NEIGH_DIST2[ki]
            coherence = coh_sum / coh_n if coh_n > 0 else 0.0
            gate = 1.0 if coherence > phase_thresh else 0.35
            drive = imp1 + wave_sum * 0.14 * gate + a * 0.3
            m_na = 1.0 / (1.0 + math.exp(-5.0 * (drive - 0.28)))
            na = na + (m_na - na) * 0.16
            k_inf = 0.25 + 0.55 * na
            k = k + (k_inf - k) * 0.12
            ca_inf = max(0.0, coherence) * (0.15 + imp1 * 0.5)
            ca = ca + (ca_inf - ca) * 0.18
            impulse = na * (1.0 - 0.72 * k) * (1.0 + 0.45 * ca)
            impulse = impulse * decay + wave_sum * 0.1 * gate * dt
            impulse += gravity_pull * gravity_g * dt + a * 0.28 * dt
            impulse = min(1.0, max(0.0, impulse))
            new_ph = phase + 0.09 + impulse * 0.32 + ca * 0.15
            new_ph = new_ph - TWO_PI * math.floor(new_ph / TWO_PI)
            out[y, x, CH_IMP] = impulse
            out[y, x, CH_PH] = new_ph
            out[y, x, CH_W] = weight
            out[y, x, CH_V] = v
            out[y, x, CH_NA] = min(1.0, na)
            out[y, x, CH_K] = min(1.0, k)
            out[y, x, CH_CA] = min(1.0, ca)
            out[y, x, CH_BW] = prev1[y, x, CH_BW] * 0.92


def _backward_dendrite_cpu(state, *, bw_rate, max_weight):
    h, w = state.shape[:2]
    for y in range(h):
        for x in range(w):
            imp = state[y, x, CH_IMP]
            phase = state[y, x, CH_PH]
            ca = state[y, x, CH_CA]
            wv = state[y, x, CH_W]
            feedback = 0.0
            for ki in range(8):
                nx, ny = x + NEIGH_DX[ki], y + NEIGH_DY[ki]
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                n_imp = state[ny, nx, CH_IMP]
                n_ph = state[ny, nx, CH_PH]
                c = math.cos(n_ph - phase)
                if c > 0.55:
                    feedback += n_imp * c * ca
            bw = feedback * 0.25
            state[y, x, CH_BW] = bw
            if bw > 0.03:
                wv = min(max_weight, wv + bw_rate * bw)
            state[y, x, CH_W] = wv
