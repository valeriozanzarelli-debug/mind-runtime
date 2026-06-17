"""Kernel CUDA ottimizzati V2 — shared memory tiles, sync batching.

Usato da gpu_physics_v2 quando CUDA è disponibile.
"""

from __future__ import annotations

import math

import numpy as np

from mindruntime.cuda_util import HAS_CUDA, HAS_NUMBA
from mindruntime.field_v2 import CH_CA, CH_COH, CH_EN, CH_IMP, CH_PH, CH_V, CH_W, PI, TWO_PI

if HAS_NUMBA:
    from numba import cuda, float32, int32
else:  # pragma: no cover
    cuda = None  # type: ignore

TILE = 16


def _cuda_grid(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    threads = (TILE, TILE)
    blocks = ((w + TILE - 1) // TILE, (h + TILE - 1) // TILE)
    return blocks, threads


if HAS_NUMBA and cuda is not None:

    @cuda.jit(device=True)
    def _wrap_mod(v, n):
        r = v % n
        if r < 0:
            r += n
        return r

    @cuda.jit(device=True)
    def _wrap_phase(dph):
        while dph > PI:
            dph -= TWO_PI
        while dph < -PI:
            dph += TWO_PI
        return dph

    @cuda.jit(device=True)
    def _cooperative_load2(field, sh0, sh1, h, w, shared_sz, ch0, ch1, halo):
        ty = cuda.threadIdx.y
        tx = cuda.threadIdx.x
        by = cuda.blockIdx.y
        bx = cuda.blockIdx.x
        bdy = cuda.blockDim.y
        bdx = cuda.blockDim.x
        for sy in range(ty, shared_sz, bdy):
            for sx in range(tx, shared_sz, bdx):
                gy = _wrap_mod(by * bdy + sy - halo, h)
                gx = _wrap_mod(bx * bdx + sx - halo, w)
                sh0[sy, sx] = field[gy, gx, ch0]
                sh1[sy, sx] = field[gy, gx, ch1]

    @cuda.jit(device=True)
    def _cooperative_load3(field, sh0, sh1, sh2, h, w, shared_sz, ch0, ch1, ch2, halo):
        ty = cuda.threadIdx.y
        tx = cuda.threadIdx.x
        by = cuda.blockIdx.y
        bx = cuda.blockIdx.x
        bdy = cuda.blockDim.y
        bdx = cuda.blockDim.x
        for sy in range(ty, shared_sz, bdy):
            for sx in range(tx, shared_sz, bdx):
                gy = _wrap_mod(by * bdy + sy - halo, h)
                gx = _wrap_mod(bx * bdx + sx - halo, w)
                sh0[sy, sx] = field[gy, gx, ch0]
                sh1[sy, sx] = field[gy, gx, ch1]
                sh2[sy, sx] = field[gy, gx, ch2]

    @cuda.jit
    def _turing_rd_cuda_opt(field_in, field_out, decay):
        halo = 1
        shared_sz = TILE + 2 * halo
        sh_imp = cuda.shared.array((18, 18), dtype=float32)
        sh_ph = cuda.shared.array((18, 18), dtype=float32)

        _cooperative_load2(field_in, sh_imp, sh_ph, field_in.shape[0], field_in.shape[1], shared_sz, 0, 1, halo)
        cuda.syncthreads()

        x = cuda.blockIdx.x * TILE + cuda.threadIdx.x
        y = cuda.blockIdx.y * TILE + cuda.threadIdx.y
        h, w = field_in.shape[0], field_in.shape[1]
        if x >= w or y >= h:
            return

        sx = cuda.threadIdx.x + halo
        sy = cuda.threadIdx.y + halo
        impulse = sh_imp[sy, sx]
        phase = sh_ph[sy, sx]
        calcium = field_in[y, x, 2]
        voltage = field_in[y, x, 5]
        reaction = 0.5 * (1.0 + calcium * 2.0) if voltage > 0.0 else 0.0
        dc, dd = 0.0, 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                ni = sh_imp[sy + dy, sx + dx]
                np_ = sh_ph[sy + dy, sx + dx]
                dph = _wrap_phase(np_ - phase)
                dist = math.sqrt(dx * dx + dy * dy)
                if abs(dph) < PI / 4:
                    dc += ni * math.cos(dph) / (1.0 + dist)
                else:
                    dd += ni * math.sin(dph) / (2.0 + dist)
        imp_n = max(0.0, min(1.0, (impulse + reaction + dc - dd) * decay))
        ph_n = (phase + 0.25 + calcium * 0.6) % TWO_PI
        field_out[y, x, 0] = imp_n
        field_out[y, x, 1] = ph_n
        for ch in range(2, 12):
            field_out[y, x, ch] = field_in[y, x, ch]

    @cuda.jit
    def _gamma_lock_cuda_opt(field, threshold):
        halo = 3
        shared_sz = TILE + 2 * halo
        sh_ph = cuda.shared.array((22, 22), dtype=float32)

        ty = cuda.threadIdx.y
        tx = cuda.threadIdx.x
        by = cuda.blockIdx.y
        bx = cuda.blockIdx.x
        bdy = cuda.blockDim.y
        bdx = cuda.blockDim.x
        h, w = field.shape[0], field.shape[1]
        for sy in range(ty, shared_sz, bdy):
            for sx in range(tx, shared_sz, bdx):
                gy = _wrap_mod(by * bdy + sy - halo, h)
                gx = _wrap_mod(bx * bdx + sx - halo, w)
                sh_ph[sy, sx] = field[gy, gx, 1]
        cuda.syncthreads()

        x = bx * TILE + tx
        y = by * TILE + ty
        if x >= w or y >= h:
            return

        sx = tx + halo
        sy = ty + halo
        sc, ss, n = 0.0, 0.0, 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ph = sh_ph[sy + dy, sx + dx]
                sc += math.cos(ph)
                ss += math.sin(ph)
                n += 1
        R = math.sqrt(sc * sc + ss * ss) / max(1, n)
        coh = field[y, x, 8]
        if R > threshold:
            coh = min(1.0, coh + 0.05)
            mean_ph = math.atan2(ss, sc)
            cur = field[y, x, 1]
            delta = _wrap_phase(mean_ph - cur)
            field[y, x, 1] = (cur + 0.1 * delta) % TWO_PI
        else:
            coh *= 0.95
        field[y, x, 8] = max(0.0, min(1.0, coh))

    @cuda.jit
    def _soc_cuda_opt(field, target):
        halo = 5
        shared_sz = TILE + 2 * halo
        sh_imp = cuda.shared.array((26, 26), dtype=float32)
        sh_v = cuda.shared.array((26, 26), dtype=float32)

        _cooperative_load2(field, sh_imp, sh_v, field.shape[0], field.shape[1], shared_sz, 0, 5, halo)
        cuda.syncthreads()

        x = cuda.blockIdx.x * TILE + cuda.threadIdx.x
        y = cuda.blockIdx.y * TILE + cuda.threadIdx.y
        h, w = field.shape[0], field.shape[1]
        if x >= w or y >= h:
            return

        sx = cuda.threadIdx.x + halo
        sy = cuda.threadIdx.y + halo
        active = 0
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if sh_imp[sy + dy, sx + dx] > 0.5 and sh_v[sy + dy, sx + dx] > -30.0:
                    active += 1
        wv = field[y, x, 6]
        if active > target:
            wv -= 0.01
        elif active < target:
            wv += 0.01
        field[y, x, 6] = max(0.1, min(1.0, wv))

    @cuda.jit
    def _predictive_cuda_opt(field, spike_time, tick, noise):
        halo = 1
        shared_sz = TILE + 2 * halo
        sh_imp = cuda.shared.array((18, 18), dtype=float32)
        sh_w = cuda.shared.array((18, 18), dtype=float32)

        _cooperative_load2(field, sh_imp, sh_w, field.shape[0], field.shape[1], shared_sz, 0, 6, halo)
        cuda.syncthreads()

        x = cuda.blockIdx.x * TILE + cuda.threadIdx.x
        y = cuda.blockIdx.y * TILE + cuda.threadIdx.y
        h, w = field.shape[0], field.shape[1]
        if x >= w or y >= h:
            return

        sx = cuda.threadIdx.x + halo
        sy = cuda.threadIdx.y + halo
        pred, tw = 0.0, 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                wt = sh_w[sy + dy, sx + dx]
                pred += sh_imp[sy + dy, sx + dx] * wt
                tw += wt
        if tw > 0:
            pred /= tw
        actual = field[y, x, 0]
        err = abs(actual - pred)
        coh = field[y, x, 8]
        if err > 0.3:
            idx = y * w + x
            st = (int(tick) + idx * 1103515245 + 12345) & 0x7FFFFFFF
            r = (st % 10000) / 10000.0
            wv = field[y, x, 6] + (r - 0.5) * noise
            field[y, x, 6] = max(0.1, min(1.0, wv))
        if err < 0.1 and coh > 0.7:
            en = field[y, x, 7] + 0.01
            field[y, x, 7] = min(1.0, en)
        if field[y, x, 5] > 0.0:
            spike_time[y, x] = tick

    @cuda.jit
    def _inject_rgb_cuda(field, rgb, gain):
        y, x = cuda.grid(2)
        h, w = field.shape[0], field.shape[1]
        rh, rw = rgb.shape[0], rgb.shape[1]
        if y >= h or y >= rh or x >= w or x >= rw:
            return
        lum = 0.299 * rgb[y, x, 0] + 0.587 * rgb[y, x, 1] + 0.114 * rgb[y, x, 2]
        if lum > 0.14:
            imp = field[y, x, 0] + lum * gain
            field[y, x, 0] = min(1.0, imp)
            na = field[y, x, 3] + lum * gain * 0.3
            field[y, x, 3] = min(1.0, na)


def launch_turing(field_in: np.ndarray, field_out: np.ndarray, decay: float) -> None:
    h, w = field_in.shape[:2]
    blocks, threads = _cuda_grid(h, w)
    _turing_rd_cuda_opt[blocks, threads](field_in, field_out, float32(decay))


def launch_gamma(field: np.ndarray, threshold: float) -> None:
    h, w = field.shape[:2]
    blocks, threads = _cuda_grid(h, w)
    _gamma_lock_cuda_opt[blocks, threads](field, float32(threshold))


def launch_soc(field: np.ndarray, target: int) -> None:
    h, w = field.shape[:2]
    blocks, threads = _cuda_grid(h, w)
    _soc_cuda_opt[blocks, threads](field, int32(target))


def launch_predictive(field: np.ndarray, spike_time: np.ndarray, tick: float, noise: float) -> None:
    h, w = field.shape[:2]
    blocks, threads = _cuda_grid(h, w)
    _predictive_cuda_opt[blocks, threads](field, spike_time, float32(tick), float32(noise))


def launch_inject_rgb(field: np.ndarray, rgb: np.ndarray, gain: float) -> None:
    rgb = np.ascontiguousarray(rgb.astype(np.float32))
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    h, w = field.shape[:2]
    blocks, threads = _cuda_grid(h, min(w, rgb.shape[1]))
    _inject_rgb_cuda[blocks, threads](field, rgb, float32(gain))


def cuda_sync() -> None:
    if HAS_CUDA and cuda is not None:
        cuda.synchronize()


def optimized_available() -> bool:
    return bool(HAS_CUDA and cuda is not None)
