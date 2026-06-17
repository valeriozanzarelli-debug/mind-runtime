"""Kernel fisica V2 — Hodgkin-Huxley RK4, Turing, SOC, gamma, predictive coding.

Compatibile CPU (test/CI) e Numba CUDA (RTX 1060 locale).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mindruntime.cuda_util import HAS_CUDA, HAS_NUMBA
from mindruntime.field_v2 import (
    CH_CA,
    CH_COH,
    CH_EN,
    CH_H,
    CH_IMP,
    CH_K,
    CH_M,
    CH_N,
    CH_NA,
    CH_PH,
    CH_V,
    CH_W,
    N_CH,
    PI,
    TWO_PI,
)

if HAS_NUMBA:
    from numba import cuda, float32, int32
else:  # pragma: no cover
    cuda = None  # type: ignore

# Hodgkin-Huxley (unità normalizzate per simulazione real-time)
G_NA, G_K, G_L = 120.0, 36.0, 0.3
E_NA, E_K, E_L = 50.0, -77.0, -54.387
C_M = 1.0
HH_DT = 0.08


def _lcg(seed: int, idx: int) -> float:
    st = (seed + idx * 1103515245 + 12345) & 0x7FFFFFFF
    return (st % 10000) / 10000.0


# --- Hodgkin-Huxley helpers (CPU) ---

def alpha_m(V: float) -> float:
    x = V + 40.0
    if abs(x) < 1e-6:
        return 1.0
    return 0.1 * x / (1.0 - math.exp(-x / 10.0))


def beta_m(V: float) -> float:
    return 4.0 * math.exp(-(V + 65.0) / 18.0)


def alpha_h(V: float) -> float:
    return 0.07 * math.exp(-(V + 65.0) / 20.0)


def beta_h(V: float) -> float:
    return 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))


def alpha_n(V: float) -> float:
    x = V + 55.0
    if abs(x) < 1e-6:
        return 0.1
    return 0.01 * x / (1.0 - math.exp(-x / 10.0))


def beta_n(V: float) -> float:
    return 0.125 * math.exp(-(V + 65.0) / 80.0)


def _dV_dt(V: float, m: float, h: float, n: float, I_ext: float) -> float:
    I_Na = G_NA * (m**3) * h * (E_NA - V)
    I_K = G_K * (n**4) * (E_K - V)
    I_L = G_L * (E_L - V)
    return (I_Na + I_K + I_L + I_ext) / C_M


def hh_rk4_step_field(field: np.ndarray, dt: float = HH_DT) -> None:
    h, w = field.shape[:2]
    for y in range(h):
        for x in range(w):
            V = float(field[y, x, CH_V])
            m = float(field[y, x, CH_M])
            hv = float(field[y, x, CH_H])
            n = float(field[y, x, CH_N])
            I_ext = float(field[y, x, CH_IMP]) * 10.0

            def dm(Vv, mv):
                return alpha_m(Vv) * (1 - mv) - beta_m(Vv) * mv

            def dh(Vv, hv_):
                return alpha_h(Vv) * (1 - hv_) - beta_h(Vv) * hv_

            def dn(Vv, nv):
                return alpha_n(Vv) * (1 - nv) - beta_n(Vv) * nv

            k1_V = _dV_dt(V, m, hv, n, I_ext)
            k1_m, k1_h, k1_n = dm(V, m), dh(V, hv), dn(V, n)

            k2_V = _dV_dt(V + 0.5 * dt * k1_V, m + 0.5 * dt * k1_m, hv + 0.5 * dt * k1_h, n + 0.5 * dt * k1_n, I_ext)
            k2_m = dm(V + 0.5 * dt * k1_V, m + 0.5 * dt * k1_m)
            k2_h = dh(V + 0.5 * dt * k1_V, hv + 0.5 * dt * k1_h)
            k2_n = dn(V + 0.5 * dt * k1_V, n + 0.5 * dt * k1_n)

            k3_V = _dV_dt(V + 0.5 * dt * k2_V, m + 0.5 * dt * k2_m, hv + 0.5 * dt * k2_h, n + 0.5 * dt * k2_n, I_ext)
            k3_m = dm(V + 0.5 * dt * k2_V, m + 0.5 * dt * k2_m)
            k3_h = dh(V + 0.5 * dt * k2_V, hv + 0.5 * dt * k2_h)
            k3_n = dn(V + 0.5 * dt * k2_V, n + 0.5 * dt * k2_n)

            k4_V = _dV_dt(V + dt * k3_V, m + dt * k3_m, hv + dt * k3_h, n + dt * k3_n, I_ext)
            k4_m = dm(V + dt * k3_V, m + dt * k3_m)
            k4_h = dh(V + dt * k3_V, hv + dt * k3_h)
            k4_n = dn(V + dt * k3_V, n + dt * k3_n)

            Vn = V + (dt / 6.0) * (k1_V + 2 * k2_V + 2 * k3_V + k4_V)
            mn = np.clip(m + (dt / 6.0) * (k1_m + 2 * k2_m + 2 * k3_m + k4_m), 0, 1)
            hn = np.clip(hv + (dt / 6.0) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h), 0, 1)
            nn = np.clip(n + (dt / 6.0) * (k1_n + 2 * k2_n + 2 * k3_n + k4_n), 0, 1)

            field[y, x, CH_V] = Vn
            field[y, x, CH_M] = mn
            field[y, x, CH_H] = hn
            field[y, x, CH_N] = nn

            if Vn > 0.0:
                field[y, x, CH_CA] = min(1.0, float(field[y, x, CH_CA]) + 0.002)
            else:
                field[y, x, CH_CA] *= 0.992


def initialize_field_v2(rgb: np.ndarray, field: np.ndarray, *, seed: int = 42, threshold: float = 0.12) -> None:
    rgb = np.ascontiguousarray(rgb.astype(np.float32))
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    h, w = field.shape[:2]
    for y in range(h):
        for x in range(w):
            lum = 0.299 * rgb[y, x, 0] + 0.587 * rgb[y, x, 1] + 0.114 * rgb[y, x, 2]
            r1, r2 = _lcg(seed, y * w + x), _lcg(seed + 7, y * w + x)
            field[y, x, CH_IMP] = float(lum) if lum > threshold else 0.0
            field[y, x, CH_PH] = r1 * TWO_PI
            field[y, x, CH_CA] = 0.05
            field[y, x, CH_NA] = 0.08
            field[y, x, CH_K] = 0.55
            field[y, x, CH_V] = -70.0
            field[y, x, CH_W] = 0.45 + (r2 - 0.5) * 0.15
            field[y, x, CH_EN] = 0.4
            field[y, x, CH_COH] = 0.0
            V = -70.0
            am, ah, an = alpha_m(V), alpha_h(V), alpha_n(V)
            field[y, x, CH_M] = am / (am + beta_m(V) + 1e-9)
            field[y, x, CH_H] = ah / (ah + beta_h(V) + 1e-9)
            field[y, x, CH_N] = an / (an + beta_n(V) + 1e-9)


def turing_reaction_diffusion(field_in: np.ndarray, field_out: np.ndarray, *, decay: float = 0.97) -> None:
    h, w = field_in.shape[:2]
    for y in range(h):
        for x in range(w):
            impulse = float(field_in[y, x, CH_IMP])
            phase = float(field_in[y, x, CH_PH])
            calcium = float(field_in[y, x, CH_CA])
            voltage = float(field_in[y, x, CH_V])
            reaction = 0.0
            if voltage > 0.0:
                reaction = 0.5 * (1.0 + calcium * 2.0)
            dc, dd = 0.0, 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = (y + dy) % h, (x + dx) % w
                    ni = float(field_in[ny, nx, CH_IMP])
                    np_ = float(field_in[ny, nx, CH_PH])
                    dph = np_ - phase
                    while dph > PI:
                        dph -= TWO_PI
                    while dph < -PI:
                        dph += TWO_PI
                    dist = math.sqrt(dx * dx + dy * dy)
                    if abs(dph) < PI / 4:
                        dc += ni * math.cos(dph) / (1.0 + dist)
                    else:
                        dd += ni * math.sin(dph) / (2.0 + dist)
            imp_n = np.clip((impulse + reaction + dc - dd) * decay, 0, 1)
            omega = 0.25 + calcium * 0.6
            ph_n = (phase + omega) % TWO_PI
            field_out[y, x, CH_IMP] = imp_n
            field_out[y, x, CH_PH] = ph_n
            for ch in range(2, N_CH):
                field_out[y, x, ch] = field_in[y, x, ch]


def soc_avalanche(field: np.ndarray, *, target: int = 10) -> None:
    h, w = field.shape[:2]
    radius = 5
    for y in range(h):
        for x in range(w):
            active = 0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = (y + dy) % h, (x + dx) % w
                    if field[ny, nx, CH_IMP] > 0.5 and field[ny, nx, CH_V] > -30.0:
                        active += 1
            wv = float(field[y, x, CH_W])
            if active > target:
                wv -= 0.01
            elif active < target:
                wv += 0.01
            field[y, x, CH_W] = np.clip(wv, 0.1, 1.0)


def gamma_phase_lock(field: np.ndarray, *, threshold: float = 0.65) -> None:
    h, w = field.shape[:2]
    radius = 3
    for y in range(h):
        for x in range(w):
            sc, ss, n = 0.0, 0.0, 0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = (y + dy) % h, (x + dx) % w
                    ph = float(field[ny, nx, CH_PH])
                    sc += math.cos(ph)
                    ss += math.sin(ph)
                    n += 1
            R = math.sqrt(sc * sc + ss * ss) / max(1, n)
            coh = float(field[y, x, CH_COH])
            if R > threshold:
                coh = min(1.0, coh + 0.05)
                mean_ph = math.atan2(ss, sc)
                cur = float(field[y, x, CH_PH])
                delta = mean_ph - cur
                while delta > PI:
                    delta -= TWO_PI
                while delta < -PI:
                    delta += TWO_PI
                field[y, x, CH_PH] = (cur + 0.1 * delta) % TWO_PI
            else:
                coh *= 0.95
            field[y, x, CH_COH] = np.clip(coh, 0, 1)


def predictive_coding(field: np.ndarray, spike_time: np.ndarray, tick: float, *, noise: float = 0.02) -> float:
    """Ritorna free energy media."""
    h, w = field.shape[:2]
    err_sum = 0.0
    for y in range(h):
        for x in range(w):
            pred, tw = 0.0, 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = (y + dy) % h, (x + dx) % w
                    wt = float(field[ny, nx, CH_W])
                    pred += float(field[ny, nx, CH_IMP]) * wt
                    tw += wt
            if tw > 0:
                pred /= tw
            actual = float(field[y, x, CH_IMP])
            err = abs(actual - pred)
            err_sum += err
            coh = float(field[y, x, CH_COH])
            if err > 0.3:
                field[y, x, CH_W] = np.clip(field[y, x, CH_W] + (_lcg(int(tick), y * w + x) - 0.5) * noise, 0.1, 1.0)
            if err < 0.1 and coh > 0.7:
                field[y, x, CH_EN] = min(1.0, float(field[y, x, CH_EN]) + 0.01)
            if float(field[y, x, CH_V]) > 0.0:
                spike_time[y, x] = tick
    return err_sum / max(1, h * w)


def kuramoto_global(phase: np.ndarray) -> float:
    c, s = np.cos(phase).mean(), np.sin(phase).mean()
    return float(np.sqrt(c * c + s * s))


def inject_rgb(field: np.ndarray, rgb: np.ndarray, *, gain: float = 0.35) -> None:
    if rgb.max() > 1.5:
        rgb = rgb.astype(np.float32) / 255.0
    h, w = field.shape[:2]
    for y in range(min(h, rgb.shape[0])):
        for x in range(min(w, rgb.shape[1])):
            lum = 0.299 * rgb[y, x, 0] + 0.587 * rgb[y, x, 1] + 0.114 * rgb[y, x, 2]
            if lum > 0.14:
                field[y, x, CH_IMP] = min(1.0, float(field[y, x, CH_IMP]) + lum * gain)
                field[y, x, CH_NA] = min(1.0, float(field[y, x, CH_NA]) + lum * gain * 0.3)


# CUDA wrappers (delegano ai kernel CPU se non disponibile)
def physics_step_v2(
    field: np.ndarray,
    scratch: np.ndarray,
    spike_time: np.ndarray,
    tick: float,
    *,
    do_soc: bool,
) -> dict[str, float]:
    for _ in range(2):
        hh_rk4_step_field(field)
    turing_reaction_diffusion(field, scratch)
    field[:] = scratch
    if do_soc:
        soc_avalanche(field)
    gamma_phase_lock(field)
    fe = predictive_coding(field, spike_time, tick)
    R = kuramoto_global(field[:, :, CH_PH])
    return {"order": R, "free_energy": fe, "mean_coherence": float(field[:, :, CH_COH].mean())}
