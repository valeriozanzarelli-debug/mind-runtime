"""Kernel Numba CUDA — fisica emergente: Hodgkin-Huxley, Turing, SOC, biforcazione.

Texture neurone per pixel (8 canali):
  0 impulse   1 phase   2 calcium   3 sodium   4 potassium
  5 weight    6 energy  7 coherence

Stato Hodgkin-Huxley separato (4): m, h, n, V (mV)
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
N_CHANNELS = 8
CH_IMP, CH_PH, CH_CA, CH_NA, CH_K, CH_W, CH_EN, CH_COH = 0, 1, 2, 3, 4, 5, 6, 7
HH_M, HH_H, HH_N, HH_V = 0, 1, 2, 3

NEIGH_DX = (-1, 0, 1, -1, 1, -1, 0, 1)
NEIGH_DY = (-1, -1, -1, 0, 0, 1, 1, 1)
NEIGH_DIST = (1.4142135, 1.0, 1.4142135, 1.0, 1.0, 1.4142135, 1.0, 1.4142135)

# Costanti HH (mV, ms)
_C_M = 1.0
_G_NA = 120.0
_G_K = 36.0
_G_L = 0.3
_E_NA = 50.0
_E_K = -77.0
_E_L = -54.3


# --- funzioni device HH ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit(device=True)
    def _safe_exp(x):
        if x < -40.0:
            return 0.0
        if x > 40.0:
            return math.exp(40.0)
        return math.exp(x)

    @cuda.jit(device=True)
    def _alpha_m(V):
        x = V + 40.0
        if abs(x) < 1e-6:
            return 1.0
        return 0.1 * x / (1.0 - _safe_exp(-x / 10.0))

    @cuda.jit(device=True)
    def _beta_m(V):
        return 4.0 * _safe_exp(-(V + 65.0) / 18.0)

    @cuda.jit(device=True)
    def _alpha_h(V):
        return 0.07 * _safe_exp(-(V + 65.0) / 20.0)

    @cuda.jit(device=True)
    def _beta_h(V):
        return 1.0 / (1.0 + _safe_exp(-(V + 35.0) / 10.0))

    @cuda.jit(device=True)
    def _alpha_n(V):
        x = V + 55.0
        if abs(x) < 1e-6:
            return 0.1
        return 0.01 * x / (1.0 - _safe_exp(-x / 10.0))

    @cuda.jit(device=True)
    def _beta_n(V):
        return 0.125 * _safe_exp(-(V + 65.0) / 80.0)

    @cuda.jit(device=True)
    def _hh_deriv(V, m, h, n):
        """Derivate HH: dV/dt, dm/dt, dh/dt, dn/dt."""
        am = _alpha_m(V)
        bm = _beta_m(V)
        ah = _alpha_h(V)
        bh = _beta_h(V)
        an = _alpha_n(V)
        bn = _beta_n(V)
        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n
        I_na = _G_NA * m * m * m * h * (V - _E_NA)
        I_k = _G_K * n * n * n * n * (V - _E_K)
        I_l = _G_L * (V - _E_L)
        dV = -(I_na + I_k + I_l) / _C_M
        return dV, dm, dh, dn

    @cuda.jit(device=True)
    def _hh_rk4(V, m, h, n, dt):
        """Integrazione Runge-Kutta 4° ordine per un pixel."""
        k1V, k1m, k1h, k1n = _hh_deriv(V, m, h, n)
        k2V, k2m, k2h, k2n = _hh_deriv(
            V + 0.5 * dt * k1V, m + 0.5 * dt * k1m, h + 0.5 * dt * k1h, n + 0.5 * dt * k1n
        )
        k3V, k3m, k3h, k3n = _hh_deriv(
            V + 0.5 * dt * k2V, m + 0.5 * dt * k2m, h + 0.5 * dt * k2h, n + 0.5 * dt * k2n
        )
        k4V, k4m, k4h, k4n = _hh_deriv(
            V + dt * k3V, m + dt * k3m, h + dt * k3h, n + dt * k3n
        )
        Vn = V + dt * (k1V + 2.0 * k2V + 2.0 * k3V + k4V) / 6.0
        mn = m + dt * (k1m + 2.0 * k2m + 2.0 * k3m + k4m) / 6.0
        hn = h + dt * (k1h + 2.0 * k2h + 2.0 * k3h + k4h) / 6.0
        nn = n + dt * (k1n + 2.0 * k2n + 2.0 * k3n + k4n) / 6.0
        mn = min(1.0, max(0.0, mn))
        hn = min(1.0, max(0.0, hn))
        nn = min(1.0, max(0.0, nn))
        return Vn, mn, hn, nn


# --- Kernel A: initialize_field ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _initialize_field_cuda(rgb, field, threshold, seed):
        """Pixel luminoso → impulso; ioni e pesi iniziali realistici."""
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
        st = (st * 1103515245 + 12345) & 0x7FFFFFFF
        r3 = (st % 10000) / 10000.0
        field[y, x, CH_PH] = r1 * TWO_PI
        field[y, x, CH_CA] = 0.1
        field[y, x, CH_NA] = 0.7
        field[y, x, CH_K] = 0.6
        field[y, x, CH_W] = 0.5 + (r2 - 0.5) * 0.2
        field[y, x, CH_EN] = 0.5 + (r3 - 0.5) * 0.2
        field[y, x, CH_COH] = 0.0
        field[y, x, CH_IMP] = min(1.0, lum) if lum > threshold else 0.0


# --- Kernel B: hodgkin_huxley_step ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _hodgkin_huxley_step_cuda(field, hh_state, dt, impulse_drive):
        """RK4 HH per pixel; aggiorna ioni nel campo."""
        y, x = cuda.grid(2)
        h, w = field.shape[0], field.shape[1]
        if y >= h or x >= w:
            return
        m = hh_state[y, x, HH_M]
        hh = hh_state[y, x, HH_H]
        n = hh_state[y, x, HH_N]
        V = hh_state[y, x, HH_V]
        imp = field[y, x, CH_IMP]
        # drive esterno da impulso visivo (mV)
        V = V + imp * impulse_drive
        V, m, hh, n = _hh_rk4(V, m, hh, n, dt)
        hh_state[y, x, HH_M] = m
        hh_state[y, x, HH_H] = hh
        hh_state[y, x, HH_N] = n
        hh_state[y, x, HH_V] = V
        # mappa V e gating → concentrazioni ioniche normalizzate [0,1]
        field[y, x, CH_NA] = min(1.0, max(0.0, m * hh))
        field[y, x, CH_K] = min(1.0, max(0.0, n))
        ca = field[y, x, CH_CA]
        if V > -30.0:
            ca = min(1.0, ca + 0.02 * (V + 30.0) / 80.0)
        else:
            ca = max(0.0, ca - 0.001)
        field[y, x, CH_CA] = ca
        field[y, x, CH_EN] = min(1.0, max(0.0, 0.5 + V / 200.0))


# --- Kernel C: turing_reaction_diffusion ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _turing_reaction_diffusion_cuda(
        field, prev1, prev2, hh_state, out, decay, dt, base_freq, freq_scale, react_thresh
    ):
        """Reazione-diffusione 2D con interferenza di fase."""
        y, x = cuda.grid(2)
        h, w = field.shape[0], field.shape[1]
        if y >= h or x >= w:
            return
        impulse = field[y, x, CH_IMP]
        phase = field[y, x, CH_PH]
        calcium = field[y, x, CH_CA]
        weight = field[y, x, CH_W]
        energy = field[y, x, CH_EN]
        coherence = field[y, x, CH_COH]
        V = hh_state[y, x, HH_V]

        reaction = 0.0
        if impulse > react_thresh and V > -30.0:
            reaction = 0.08 * impulse * math.cos(phase)

        diffusion = 0.0
        for k in range(8):
            nx = x + NEIGH_DX[k]
            ny = y + NEIGH_DY[k]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            n_imp = field[ny, nx, CH_IMP]
            n_ph = field[ny, nx, CH_PH]
            dph = n_ph - phase
            dist = NEIGH_DIST[k]
            if abs(dph) < 0.785398:  # π/4
                diffusion += n_imp * math.cos(dph) / (1.0 + dist)
            else:
                diffusion -= n_imp * math.sin(dph) / (1.0 + dist)

        impulse_new = impulse + diffusion * 0.12 * dt + reaction - (1.0 - decay) * impulse
        omega = base_freq + (calcium * freq_scale) % TWO_PI
        phase_new = phase + omega * dt
        phase_new = phase_new - TWO_PI * math.floor(phase_new / TWO_PI)

        out[y, x, CH_IMP] = min(1.0, max(0.0, impulse_new))
        out[y, x, CH_PH] = phase_new
        out[y, x, CH_CA] = calcium
        out[y, x, CH_NA] = field[y, x, CH_NA]
        out[y, x, CH_K] = field[y, x, CH_K]
        out[y, x, CH_W] = weight
        out[y, x, CH_EN] = energy
        out[y, x, CH_COH] = coherence


# --- Kernel D: soc_criticality_tunning ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _soc_criticality_cuda(field, avalanche_map, target_size, soc_rate, radius):
        """Auto-sintonia verso criticità (avalanche power law)."""
        y, x = cuda.grid(2)
        h, w = field.shape[0], field.shape[1]
        if y >= h or x >= w:
            return
        impulse = field[y, x, CH_IMP]
        coherence = field[y, x, CH_COH]
        weight = field[y, x, CH_W]
        local_thresh = 0.35 + weight * 0.15

        avalanche = 0.0
        if impulse > local_thresh and coherence > 0.7:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    ny, nx = y + dy, x + dx
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if field[ny, nx, CH_IMP] > local_thresh:
                        avalanche += 1.0

        avalanche_map[y, x] = avalanche
        delta = 0.0
        if avalanche > target_size:
            delta = -soc_rate * (avalanche - target_size) / target_size
        elif avalanche < target_size * 0.5:
            delta = soc_rate * (target_size * 0.5 - avalanche) / target_size
        field[y, x, CH_W] = min(2.5, max(0.02, weight + delta))


# --- Kernel E: bifurcation_phase_lock ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _bifurcation_phase_lock_cuda(field, R_thresh, lock_thresh, wave_boost):
        """Biforcazione armonica + lock-in di fase per riconoscimento."""
        y, x = cuda.grid(2)
        h, w = field.shape[0], field.shape[1]
        if y >= h or x >= w:
            return
        phase = field[y, x, CH_PH]
        impulse = field[y, x, CH_IMP]
        coherence = field[y, x, CH_COH]

        sum_cos = 0.0
        sum_sin = 0.0
        count = 0.0
        for k in range(8):
            nx = x + NEIGH_DX[k]
            ny = y + NEIGH_DY[k]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            n_ph = field[ny, nx, CH_PH]
            sum_cos += math.cos(n_ph)
            sum_sin += math.sin(n_ph)
            count += 1.0
        R = math.sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / count if count > 0 else 0.0

        if R > R_thresh:
            coherence = min(1.0, coherence + 0.04 * (R - R_thresh))
            # rompi simmetria locale
            phase = phase + 0.02 * math.sin(phase * 3.0)

        if coherence > lock_thresh:
            impulse = min(1.0, impulse + wave_boost)
            coherence = min(1.0, coherence + 0.02)

        field[y, x, CH_IMP] = impulse
        field[y, x, CH_PH] = phase - TWO_PI * math.floor(phase / TWO_PI)
        field[y, x, CH_COH] = coherence


# --- Kernel F: phase_gradient_memory ---

if HAS_NUMBA and cuda is not None:

    @cuda.jit
    def _phase_gradient_memory_cuda(field, persistence, grad_thresh, persist_needed, mem_gain):
        """Memoria emergente come gradienti di fase persistenti."""
        y, x = cuda.grid(2)
        h, w = field.shape[0], field.shape[1]
        if y >= h or x >= w:
            return
        xm = max(0, x - 1)
        xp = min(w - 1, x + 1)
        ym = max(0, y - 1)
        yp = min(h - 1, y + 1)
        grad_x = (field[y, xp, CH_PH] - field[y, xm, CH_PH]) * 0.5
        grad_y = (field[yp, x, CH_PH] - field[ym, x, CH_PH]) * 0.5
        grad_mag = math.sqrt(grad_x * grad_x + grad_y * grad_y)

        pers = persistence[y, x]
        if grad_mag > grad_thresh:
            pers = min(float(persist_needed), pers + 1.0)
        else:
            pers = max(0.0, pers - 0.5)
        persistence[y, x] = pers

        if pers >= persist_needed:
            w = field[y, x, CH_W]
            en = field[y, x, CH_EN]
            field[y, x, CH_W] = min(2.5, w + mem_gain)
            field[y, x, CH_EN] = min(1.0, en + mem_gain * 0.5)
            # risonanza: impulso vicino ad attrattore
            imp = field[y, x, CH_IMP]
            if imp > 0.1:
                field[y, x, CH_IMP] = min(1.0, imp + mem_gain * 0.3)


# --- API host ---

def _grid_2d(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    threads = (16, 16)
    blocks = ((w + 15) // 16, (h + 15) // 16)
    return blocks, threads


def initialize_field(
    rgb: np.ndarray,
    field: np.ndarray,
    *,
    threshold: float = 0.3,
    seed: int = 42,
) -> None:
    """Kernel A — inizializza texture GPU da frame RGB."""
    rgb = np.ascontiguousarray(rgb.astype(np.float32))
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    h, w = rgb.shape[:2]
    assert field.shape == (h, w, N_CHANNELS)
    if HAS_CUDA and cuda is not None:
        blocks, threads = _grid_2d(h, w)
        _initialize_field_cuda[blocks, threads](rgb, field, float32(threshold), int32(seed))
        cuda.synchronize()
    else:
        _initialize_field_cpu(rgb, field, threshold=threshold, seed=seed)


def initialize_hh_state(hh_state: np.ndarray, *, resting_v: float = -65.0) -> None:
    """Inizializza gating HH: m≈0.05, h≈0.6, n≈0.32, V resting."""
    hh_state[:, :, HH_M] = 0.05
    hh_state[:, :, HH_H] = 0.60
    hh_state[:, :, HH_N] = 0.32
    hh_state[:, :, HH_V] = resting_v


def hodgkin_huxley_step(
    field: np.ndarray,
    hh_state: np.ndarray,
    *,
    dt: float = 0.001,
    impulse_drive: float = 5.0,
) -> None:
    """Kernel B — un passo HH RK4 per ogni pixel."""
    h, w = field.shape[:2]
    assert hh_state.shape == (h, w, 4)
    if HAS_CUDA and cuda is not None:
        blocks, threads = _grid_2d(h, w)
        _hodgkin_huxley_step_cuda[blocks, threads](
            field, hh_state, float32(dt), float32(impulse_drive)
        )
        cuda.synchronize()
    else:
        _hodgkin_huxley_step_cpu(field, hh_state, dt=dt, impulse_drive=impulse_drive)


def turing_reaction_diffusion(
    field: np.ndarray,
    prev1: np.ndarray,
    prev2: np.ndarray,
    hh_state: np.ndarray,
    out: np.ndarray,
    *,
    decay: float = 0.97,
    dt: float = 1.0,
    base_freq: float = 0.08,
    freq_scale: float = 0.35,
    react_thresh: float = 0.2,
) -> None:
    """Kernel C — propagazione onda Turing."""
    h, w = field.shape[:2]
    if HAS_CUDA and cuda is not None:
        blocks, threads = _grid_2d(h, w)
        _turing_reaction_diffusion_cuda[blocks, threads](
            field,
            prev1,
            prev2,
            hh_state,
            out,
            float32(decay),
            float32(dt),
            float32(base_freq),
            float32(freq_scale),
            float32(react_thresh),
        )
        cuda.synchronize()
    else:
        _turing_reaction_diffusion_cpu(
            field, prev1, prev2, hh_state, out,
            decay=decay, dt=dt, base_freq=base_freq,
            freq_scale=freq_scale, react_thresh=react_thresh,
        )


def soc_criticality_tunning(
    field: np.ndarray,
    avalanche_map: np.ndarray,
    *,
    target_size: float = 12.0,
    soc_rate: float = 0.015,
    radius: int = 5,
) -> None:
    """Kernel D — auto-tuning SOC."""
    h, w = field.shape[:2]
    if HAS_CUDA and cuda is not None:
        blocks, threads = _grid_2d(h, w)
        _soc_criticality_cuda[blocks, threads](
            field, avalanche_map, float32(target_size), float32(soc_rate), int32(radius)
        )
        cuda.synchronize()
    else:
        _soc_criticality_cpu(
            field, avalanche_map,
            target_size=target_size, soc_rate=soc_rate, radius=radius,
        )


def bifurcation_phase_lock(
    field: np.ndarray,
    *,
    R_thresh: float = 0.6,
    lock_thresh: float = 0.8,
    wave_boost: float = 0.15,
) -> None:
    """Kernel E — biforcazione + lock-in."""
    h, w = field.shape[:2]
    if HAS_CUDA and cuda is not None:
        blocks, threads = _grid_2d(h, w)
        _bifurcation_phase_lock_cuda[blocks, threads](
            field, float32(R_thresh), float32(lock_thresh), float32(wave_boost)
        )
        cuda.synchronize()
    else:
        _bifurcation_phase_lock_cpu(
            field, R_thresh=R_thresh, lock_thresh=lock_thresh, wave_boost=wave_boost
        )


def phase_gradient_memory(
    field: np.ndarray,
    persistence: np.ndarray,
    *,
    grad_thresh: float = 0.15,
    persist_needed: float = 5.0,
    mem_gain: float = 0.012,
) -> None:
    """Kernel F — attrattori di memoria da gradiente di fase."""
    h, w = field.shape[:2]
    if HAS_CUDA and cuda is not None:
        blocks, threads = _grid_2d(h, w)
        _phase_gradient_memory_cuda[blocks, threads](
            field, persistence, float32(grad_thresh),
            float32(persist_needed), float32(mem_gain),
        )
        cuda.synchronize()
    else:
        _phase_gradient_memory_cpu(
            field, persistence,
            grad_thresh=grad_thresh, persist_needed=persist_needed, mem_gain=mem_gain,
        )


# --- CPU fallback ---

def _lcg(seed: int, idx: int) -> tuple[float, float, float]:
    st = (seed + idx * 1103515245 + 12345) & 0x7FFFFFFF
    r1 = (st % 10000) / 10000.0
    st = (st * 1103515245 + 12345) & 0x7FFFFFFF
    r2 = (st % 10000) / 10000.0
    st = (st * 1103515245 + 12345) & 0x7FFFFFFF
    r3 = (st % 10000) / 10000.0
    return r1, r2, r3


def _initialize_field_cpu(
    rgb: np.ndarray, field: np.ndarray, *, threshold: float, seed: int
) -> None:
    h, w = rgb.shape[:2]
    for y in range(h):
        for x in range(w):
            lum = 0.299 * rgb[y, x, 0] + 0.587 * rgb[y, x, 1] + 0.114 * rgb[y, x, 2]
            r1, r2, r3 = _lcg(seed, y * w + x)
            field[y, x, CH_PH] = r1 * TWO_PI
            field[y, x, CH_CA] = 0.1
            field[y, x, CH_NA] = 0.7
            field[y, x, CH_K] = 0.6
            field[y, x, CH_W] = 0.5 + (r2 - 0.5) * 0.2
            field[y, x, CH_EN] = 0.5 + (r3 - 0.5) * 0.2
            field[y, x, CH_COH] = 0.0
            field[y, x, CH_IMP] = min(1.0, lum) if lum > threshold else 0.0


def _hh_deriv_cpu(V, m, h, n):
    def safe_exp(x):
        return math.exp(max(-40.0, min(40.0, x)))

    def alpha_m(v):
        x = v + 40.0
        return 1.0 if abs(x) < 1e-6 else 0.1 * x / (1.0 - safe_exp(-x / 10.0))

    def beta_m(v):
        return 4.0 * safe_exp(-(v + 65.0) / 18.0)

    def alpha_h(v):
        return 0.07 * safe_exp(-(v + 65.0) / 20.0)

    def beta_h(v):
        return 1.0 / (1.0 + safe_exp(-(v + 35.0) / 10.0))

    def alpha_n(v):
        x = v + 55.0
        return 0.1 if abs(x) < 1e-6 else 0.01 * x / (1.0 - safe_exp(-x / 10.0))

    def beta_n(v):
        return 0.125 * safe_exp(-(v + 65.0) / 80.0)

    am, bm = alpha_m(V), beta_m(V)
    ah, bh = alpha_h(V), beta_h(V)
    an, bn = alpha_n(V), beta_n(V)
    dm = am * (1.0 - m) - bm * m
    dh = ah * (1.0 - h) - bh * h
    dn = an * (1.0 - n) - bn * n
    I_na = _G_NA * m**3 * h * (V - _E_NA)
    I_k = _G_K * n**4 * (V - _E_K)
    I_l = _G_L * (V - _E_L)
    dV = -(I_na + I_k + I_l) / _C_M
    return dV, dm, dh, dn


def _hh_rk4_cpu(V, m, h, n, dt):
    k1 = _hh_deriv_cpu(V, m, h, n)
    k2 = _hh_deriv_cpu(V + 0.5 * dt * k1[0], m + 0.5 * dt * k1[1], h + 0.5 * dt * k1[2], n + 0.5 * dt * k1[3])
    k3 = _hh_deriv_cpu(V + 0.5 * dt * k2[0], m + 0.5 * dt * k2[1], h + 0.5 * dt * k2[2], n + 0.5 * dt * k2[3])
    k4 = _hh_deriv_cpu(V + dt * k3[0], m + dt * k3[1], h + dt * k3[2], n + dt * k3[3])
    Vn = V + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
    mn = m + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
    hn = h + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
    nn = n + dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0
    return Vn, min(1.0, max(0.0, mn)), min(1.0, max(0.0, hn)), min(1.0, max(0.0, nn))


def _hodgkin_huxley_step_cpu(
    field: np.ndarray, hh_state: np.ndarray, *, dt: float, impulse_drive: float
) -> None:
    h, w = field.shape[:2]
    for y in range(h):
        for x in range(w):
            m = hh_state[y, x, HH_M]
            hh = hh_state[y, x, HH_H]
            n = hh_state[y, x, HH_N]
            V = hh_state[y, x, HH_V] + field[y, x, CH_IMP] * impulse_drive
            V, m, hh, n = _hh_rk4_cpu(V, m, hh, n, dt)
            hh_state[y, x, HH_M] = m
            hh_state[y, x, HH_H] = hh
            hh_state[y, x, HH_N] = n
            hh_state[y, x, HH_V] = V
            field[y, x, CH_NA] = min(1.0, max(0.0, m * hh))
            field[y, x, CH_K] = min(1.0, max(0.0, n))
            ca = field[y, x, CH_CA]
            if V > -30.0:
                ca = min(1.0, ca + 0.02 * (V + 30.0) / 80.0)
            else:
                ca = max(0.0, ca - 0.001)
            field[y, x, CH_CA] = ca
            field[y, x, CH_EN] = min(1.0, max(0.0, 0.5 + V / 200.0))


def _turing_reaction_diffusion_cpu(
    field, prev1, prev2, hh_state, out, **kw: Any
) -> None:
    decay = kw["decay"]
    dt = kw["dt"]
    base_freq = kw["base_freq"]
    freq_scale = kw["freq_scale"]
    react_thresh = kw["react_thresh"]
    h, w = field.shape[:2]
    for y in range(h):
        for x in range(w):
            impulse = field[y, x, CH_IMP]
            phase = field[y, x, CH_PH]
            calcium = field[y, x, CH_CA]
            weight = field[y, x, CH_W]
            energy = field[y, x, CH_EN]
            coherence = field[y, x, CH_COH]
            V = hh_state[y, x, HH_V]
            reaction = 0.08 * impulse * math.cos(phase) if impulse > react_thresh and V > -30.0 else 0.0
            diffusion = 0.0
            for k in range(8):
                nx, ny = x + NEIGH_DX[k], y + NEIGH_DY[k]
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                n_imp = field[ny, nx, CH_IMP]
                n_ph = field[ny, nx, CH_PH]
                dph = n_ph - phase
                dist = NEIGH_DIST[k]
                if abs(dph) < math.pi / 4:
                    diffusion += n_imp * math.cos(dph) / (1.0 + dist)
                else:
                    diffusion -= n_imp * math.sin(dph) / (1.0 + dist)
            impulse_new = impulse + diffusion * 0.12 * dt + reaction - (1.0 - decay) * impulse
            omega = base_freq + (calcium * freq_scale) % TWO_PI
            phase_new = (phase + omega * dt) % TWO_PI
            out[y, x, CH_IMP] = min(1.0, max(0.0, impulse_new))
            out[y, x, CH_PH] = phase_new
            out[y, x, CH_CA] = calcium
            out[y, x, CH_NA] = field[y, x, CH_NA]
            out[y, x, CH_K] = field[y, x, CH_K]
            out[y, x, CH_W] = weight
            out[y, x, CH_EN] = energy
            out[y, x, CH_COH] = coherence


def _soc_criticality_cpu(
    field, avalanche_map, *, target_size, soc_rate, radius
) -> None:
    h, w = field.shape[:2]
    for y in range(h):
        for x in range(w):
            impulse = field[y, x, CH_IMP]
            coherence = field[y, x, CH_COH]
            weight = field[y, x, CH_W]
            local_thresh = 0.35 + weight * 0.15
            avalanche = 0.0
            if impulse > local_thresh and coherence > 0.7:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx * dx + dy * dy > radius * radius:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= nx < w and 0 <= ny < h and field[ny, nx, CH_IMP] > local_thresh:
                            avalanche += 1.0
            avalanche_map[y, x] = avalanche
            delta = 0.0
            if avalanche > target_size:
                delta = -soc_rate * (avalanche - target_size) / target_size
            elif avalanche < target_size * 0.5:
                delta = soc_rate * (target_size * 0.5 - avalanche) / target_size
            field[y, x, CH_W] = min(2.5, max(0.02, weight + delta))


def _bifurcation_phase_lock_cpu(field, *, R_thresh, lock_thresh, wave_boost) -> None:
    h, w = field.shape[:2]
    for y in range(h):
        for x in range(w):
            phase = field[y, x, CH_PH]
            impulse = field[y, x, CH_IMP]
            coherence = field[y, x, CH_COH]
            sum_cos = sum_sin = count = 0.0
            for k in range(8):
                nx, ny = x + NEIGH_DX[k], y + NEIGH_DY[k]
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                n_ph = field[ny, nx, CH_PH]
                sum_cos += math.cos(n_ph)
                sum_sin += math.sin(n_ph)
                count += 1.0
            R = math.sqrt(sum_cos**2 + sum_sin**2) / count if count else 0.0
            if R > R_thresh:
                coherence = min(1.0, coherence + 0.04 * (R - R_thresh))
                phase += 0.02 * math.sin(phase * 3.0)
            if coherence > lock_thresh:
                impulse = min(1.0, impulse + wave_boost)
                coherence = min(1.0, coherence + 0.02)
            field[y, x, CH_IMP] = impulse
            field[y, x, CH_PH] = phase % TWO_PI
            field[y, x, CH_COH] = coherence


def _phase_gradient_memory_cpu(
    field, persistence, *, grad_thresh, persist_needed, mem_gain
) -> None:
    h, w = field.shape[:2]
    for y in range(h):
        for x in range(w):
            xm, xp = max(0, x - 1), min(w - 1, x + 1)
            ym, yp = max(0, y - 1), min(h - 1, y + 1)
            grad_x = (field[y, xp, CH_PH] - field[y, xm, CH_PH]) * 0.5
            grad_y = (field[yp, x, CH_PH] - field[ym, x, CH_PH]) * 0.5
            grad_mag = math.sqrt(grad_x**2 + grad_y**2)
            pers = persistence[y, x]
            if grad_mag > grad_thresh:
                pers = min(persist_needed, pers + 1.0)
            else:
                pers = max(0.0, pers - 0.5)
            persistence[y, x] = pers
            if pers >= persist_needed:
                field[y, x, CH_W] = min(2.5, field[y, x, CH_W] + mem_gain)
                field[y, x, CH_EN] = min(1.0, field[y, x, CH_EN] + mem_gain * 0.5)
                imp = field[y, x, CH_IMP]
                if imp > 0.1:
                    field[y, x, CH_IMP] = min(1.0, imp + mem_gain * 0.3)
