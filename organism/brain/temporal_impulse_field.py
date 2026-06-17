"""Campo impulsi temporale — fase locale, triple-buffer, risonanza, gravità.

Non ragiona a frame globale: ogni pixel ha tempo interno (fase + omega).
La memoria emerge da velocità/accelerazione e da attrattori gravitazionali.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

from organism.brain.gpu_backend import HAS_TORCH, resolve_device, torch_device
from organism.brain.impulse_field import (
    REGIONS,
    ImpulseBlob,
    _EXCITE,
    _INHIBIT,
)
from organism.brain.resonance_templates import SYMBOLS, build_template_bank, correlate_template

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False

if HAS_TORCH:
    import torch
    import torch.nn.functional as F

LEAK = 0.045
INERTIA = 0.48
WAVE_SPEED = 0.62
WAVE_DECAY = 0.12
PHASE_DRIVE = 0.38
SYNAPSE_GAIN = 0.34
GRAVITY_G = 0.09
PLASTICITY = 0.018
RADIAL_SIGMA = 2.2

# Frequenze banda cerebrale (Hz normalizzati per tick locale)
BAND_OMEGA = {
    "visual": 0.14,  # ~alfa
    "auditory": 0.22,  # ~beta
    "associative": 0.31,  # ~gamma bassa
    "memory": 0.18,
    "motor": 0.26,
}


@dataclass
class TemporalImpulseField:
    """Pixel-neuroni con tempo locale, onde e attrattori."""

    width: int
    height: int
    device: str = "auto"
    energy: Any = field(init=False, repr=False)
    energy_t1: Any = field(init=False, repr=False)
    energy_t2: Any = field(init=False, repr=False)
    phase: Any = field(init=False, repr=False)
    omega: Any = field(init=False, repr=False)
    mass: Any = field(init=False, repr=False)
    potential: Any = field(init=False, repr=False)
    vx: Any = field(init=False, repr=False)
    vy: Any = field(init=False, repr=False)
    trace: Any = field(init=False, repr=False)
    tick: float = 0.0
    _backend: str = field(init=False, repr=False)
    _torch_dev: Any = field(init=False, repr=False, default=None)
    _region_masks: dict[str, Any] = field(init=False, repr=False)
    _kernel_exc: Any = field(init=False, repr=False)
    _kernel_inh: Any = field(init=False, repr=False)
    _kernel_radial: Any = field(init=False, repr=False)
    _template_bank: dict[str, Any] = field(init=False, repr=False)
    _last_recognition: list[tuple[str, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width < 16 or self.height < 16:
            raise ValueError("campo temporale troppo piccolo")
        self._backend = resolve_device(self.device)
        h, w = self.height, self.width
        self._last_recognition = []
        self._template_bank = build_template_bank()["templates"]

        if self._backend != "numpy" and HAS_TORCH:
            dev = torch_device(self._backend)
            self._torch_dev = dev
            z = lambda: torch.zeros((h, w), device=dev, dtype=torch.float32)
            self.energy = z()
            self.energy_t1 = z()
            self.energy_t2 = z()
            self.phase = z()
            self.mass = z() + 0.02
            self.potential = z()
            self.vx = z()
            self.vy = z()
            self.trace = z()
            exc = torch.tensor(_EXCITE, device=dev, dtype=torch.float32)
            inh = torch.tensor(_INHIBIT, device=dev, dtype=torch.float32)
            self._kernel_exc = exc.view(1, 1, 3, 3)
            self._kernel_inh = inh.view(1, 1, 3, 3)
            self._kernel_radial = self._radial_kernel_torch(dev)
            self._region_masks = {n: self._mask_tensor(*box) for n, box in REGIONS.items()}
            self.omega = self._init_omega_torch()
        elif HAS_NUMPY and np is not None:
            self._torch_dev = None
            z = lambda: np.zeros((h, w), dtype=np.float32)
            self.energy = z()
            self.energy_t1 = z()
            self.energy_t2 = z()
            self.phase = z()
            self._region_masks = {n: self._mask_numpy(*box) for n, box in REGIONS.items()}
            self.omega = self._init_omega_numpy()
            self.mass = z() + 0.02
            self.potential = z()
            self.vx = z()
            self.vy = z()
            self.trace = z()
            self._kernel_exc = np.array(_EXCITE, dtype=np.float32)
            self._kernel_inh = np.array(_INHIBIT, dtype=np.float32)
            self._kernel_radial = self._radial_kernel_numpy()
        else:
            raise RuntimeError("TemporalImpulseField richiede numpy o torch")

    @property
    def neuron_count(self) -> int:
        return self.width * self.height

    @property
    def uses_gpu(self) -> bool:
        return self._backend.startswith("cuda")

    @property
    def temporal(self) -> bool:
        return True

    def inject_region(
        self,
        region: str,
        intensity: float = 0.7,
        *,
        pattern: list[list[float]] | None = None,
    ) -> None:
        mask = self._region_masks.get(region)
        if mask is None:
            return
        if pattern and self._is_torch():
            ph, pw = len(pattern), len(pattern[0]) if pattern else 0
            if ph > 0 and pw > 0:
                patch = torch.tensor(pattern, device=self._torch_dev, dtype=torch.float32)
                patch = patch / max(1e-6, float(patch.max()))
                y0, y1, x0, x1 = self._region_slice(region)
                th, tw = y1 - y0, x1 - x0
                if th > 0 and tw > 0:
                    scaled = F.interpolate(
                        patch.unsqueeze(0).unsqueeze(0),
                        size=(th, tw),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                    m = mask[y0:y1, x0:x1]
                    self.potential[y0:y1, x0:x1] = torch.minimum(
                        torch.tensor(1.0, device=self._torch_dev),
                        self.potential[y0:y1, x0:x1] + scaled * intensity * m,
                    )
                    self._phase_kick_region(y0, y1, x0, x1, intensity * 0.6)
                    return
        self._add_masked(self.potential, mask, intensity)
        y0, y1, x0, x1 = self._region_slice(region)
        self._phase_kick_region(y0, y1, x0, x1, intensity * 0.35)

    def inject_pixels(self, gray: list[list[float]] | list[list[int]] | list[int], *, gain: float = 0.9) -> None:
        grid = self._normalize_grid(gray)
        if not grid:
            return
        gh, gw = len(grid), len(grid[0]) if grid else 0
        y0, y1, x0, x1 = self._region_slice("visual")
        th, tw = y1 - y0, x1 - x0
        if th <= 0 or tw <= 0:
            return
        if self._is_torch():
            patch = torch.tensor(grid, device=self._torch_dev, dtype=torch.float32)
            scaled = F.interpolate(
                patch.unsqueeze(0).unsqueeze(0),
                size=(th, tw),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            m = self._region_masks["visual"][y0:y1, x0:x1]
            self.potential[y0:y1, x0:x1] = torch.minimum(
                torch.tensor(1.0, device=self._torch_dev),
                self.potential[y0:y1, x0:x1] + scaled * gain * m,
            )
            self._phase_kick_region(y0, y1, x0, x1, gain * 0.5)
        elif HAS_NUMPY and np is not None:
            patch = np.array([row[:gw] for row in grid[:gh]], dtype=np.float32)
            ys = np.linspace(0, max(0, gh - 1), th).astype(int)
            xs = np.linspace(0, max(0, gw - 1), tw).astype(int)
            sampled = patch[ys][:, xs]
            m = self._region_masks["visual"][y0:y1, x0:x1]
            self.potential[y0:y1, x0:x1] = np.minimum(
                1.0, self.potential[y0:y1, x0:x1] + sampled * gain * m
            )
            self._phase_kick_region(y0, y1, x0, x1, gain * 0.5)

    def inject_audio_band(self, bands: list[float], *, gain: float = 0.5) -> None:
        if not bands:
            return
        for i, b in enumerate(bands[:8]):
            row, col = i % 4, i // 4
            x = int(self.width * (0.05 + col * 0.08))
            y = int(self.height * (0.35 + row * 0.08))
            self._inject_point(x, y, float(b) * gain, phase_kick=float(b) * 0.8)

    def inject_text_energy(self, dim_energies: list[float], *, gain: float = 0.45) -> None:
        if not dim_energies:
            return
        y0, y1, x0, x1 = self._region_slice("associative")
        th, tw = y1 - y0, x1 - x0
        if th <= 0 or tw <= 0:
            return
        flat = dim_energies[:64]
        side = max(1, int(len(flat) ** 0.5))
        while side * side < len(flat):
            side += 1
        padded = flat + [0.0] * (side * side - len(flat))
        if self._is_torch():
            t = torch.tensor(padded, device=self._torch_dev, dtype=torch.float32).view(side, side)
            t = t / max(1e-6, float(t.max()))
            scaled = F.interpolate(
                t.unsqueeze(0).unsqueeze(0),
                size=(th, tw),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            self.potential[y0:y1, x0:x1] = torch.minimum(
                torch.tensor(1.0, device=self._torch_dev),
                self.potential[y0:y1, x0:x1] + scaled * gain,
            )
            self._phase_kick_region(y0, y1, x0, x1, gain * 0.4)
        elif HAS_NUMPY and np is not None:
            patch = np.array(padded, dtype=np.float32).reshape(side, side)
            patch = patch / max(1e-6, float(patch.max()))
            ys = np.linspace(0, side - 1, th).astype(int)
            xs = np.linspace(0, side - 1, tw).astype(int)
            scaled = patch[ys][:, xs]
            self.potential[y0:y1, x0:x1] = np.minimum(
                1.0, self.potential[y0:y1, x0:x1] + scaled * gain
            )
            self._phase_kick_region(y0, y1, x0, x1, gain * 0.4)

    def inject_memory_echo(self, signature: list[float], *, gain: float = 0.25) -> None:
        if not signature:
            return
        y0, y1, x0, x1 = self._region_slice("memory")
        n = min(len(signature), (y1 - y0) * (x1 - x0))
        if n <= 0:
            return
        side = int(n**0.5)
        if self._is_torch():
            t = torch.tensor(signature[: side * side], device=self._torch_dev, dtype=torch.float32).view(side, side)
            th, tw = y1 - y0, x1 - x0
            scaled = F.interpolate(
                t.unsqueeze(0).unsqueeze(0),
                size=(th, tw),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            self.potential[y0:y1, x0:x1] += scaled * gain
            self._phase_kick_region(y0, y1, x0, x1, gain * 0.3)
        elif HAS_NUMPY and np is not None:
            patch = np.array(signature[: side * side], dtype=np.float32).reshape(side, side)
            th, tw = y1 - y0, x1 - x0
            ys = np.linspace(0, side - 1, th).astype(int)
            xs = np.linspace(0, side - 1, tw).astype(int)
            self.potential[y0:y1, x0:x1] += patch[ys][:, xs] * gain
            self._phase_kick_region(y0, y1, x0, x1, gain * 0.3)

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self.tick += 1.0
            if self._is_torch():
                self._step_torch()
            else:
                self._step_numpy()
            if int(self.tick) % 6 == 0:
                self._last_recognition = self._recognize_symbols()

    def regional_energy(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, mask in self._region_masks.items():
            if self._is_torch():
                m = mask > 0
                e = self.energy[m]
                out[name] = float(e.mean().item()) if e.numel() else 0.0
            else:
                m = mask > 0
                out[name] = float(self.energy[m].mean()) if m.any() else 0.0
        return out

    def flux_magnitude(self) -> float:
        vel = self._velocity_field()
        if self._is_torch():
            return float(torch.sqrt(vel * vel).mean().item())
        return float(np.sqrt(vel * vel).mean())

    def acceleration_magnitude(self) -> float:
        acc = self._acceleration_field()
        if self._is_torch():
            return float(torch.sqrt(acc * acc).mean().item())
        return float(np.sqrt(acc * acc).mean())

    def phase_coherence(self) -> float:
        """Quanto le fasi locali sono allineate (attenzione emergente)."""
        if self._is_torch():
            c = torch.cos(self.phase)
            s = torch.sin(self.phase)
            mean_c = c.mean()
            mean_s = s.mean()
            r = torch.sqrt(mean_c * mean_c + mean_s * mean_s)
            return float(r.item())
        c = np.cos(self.phase)
        s = np.sin(self.phase)
        return float(np.sqrt(c.mean() ** 2 + s.mean() ** 2))

    def recognized_symbols(self) -> list[tuple[str, float]]:
        return list(self._last_recognition)

    def active_blobs(self, k: int = 12) -> list[ImpulseBlob]:
        sal = self._salience()
        blobs: list[ImpulseBlob] = []
        if self._is_torch():
            flat = sal.reshape(-1)
            vals, idxs = torch.topk(flat, min(k * 6, flat.numel()))
            taken = torch.zeros(sal.shape, dtype=torch.bool, device=self._torch_dev)
            for idx_t, val_t in zip(idxs.tolist(), vals.tolist()):
                if len(blobs) >= k or val_t < 0.035:
                    break
                y, x = divmod(int(idx_t), self.width)
                if taken[y, x]:
                    continue
                blobs.append(
                    ImpulseBlob(
                        x=x / self.width,
                        y=y / self.height,
                        energy=float(val_t),
                        vx=float(self.vx[y, x].item()),
                        vy=float(self.vy[y, x].item()),
                        region=self._point_region(x / self.width, y / self.height),
                    )
                )
                y0, y1 = max(0, y - 5), min(self.height, y + 6)
                x0, x1 = max(0, x - 5), min(self.width, x + 6)
                taken[y0:y1, x0:x1] = True
        else:
            order = np.argsort(sal.ravel())[::-1]
            taken = np.zeros(sal.shape, dtype=bool)
            for idx in order:
                if len(blobs) >= k:
                    break
                y, x = divmod(int(idx), self.width)
                v = float(sal[y, x])
                if v < 0.035:
                    break
                if taken[y, x]:
                    continue
                blobs.append(
                    ImpulseBlob(
                        x=x / self.width,
                        y=y / self.height,
                        energy=v,
                        vx=float(self.vx[y, x]),
                        vy=float(self.vy[y, x]),
                        region=self._point_region(x / self.width, y / self.height),
                    )
                )
                taken[max(0, y - 5) : y + 6, max(0, x - 5) : x + 6] = True
        return blobs

    def signature(self) -> list[float]:
        if self._is_torch():
            mix = (self.energy * 0.6 + torch.sin(self.phase) * 0.2 + self.mass * 0.2).detach().cpu().flatten()
            step = max(1, mix.numel() // 128)
            return [float(v) for v in mix[::step][:128]]
        mix = self.energy * 0.6 + np.sin(self.phase) * 0.2 + self.mass * 0.2
        flat = mix.ravel()
        step = max(1, len(flat) // 128)
        return [float(v) for v in flat[::step][:128]]

    def to_energy_bytes(self) -> bytes:
        grid = self._energy_cpu()
        out = bytearray(self.width * self.height)
        for y, row in enumerate(grid):
            for x, v in enumerate(row):
                out[y * self.width + x] = max(0, min(255, int(v * 255)))
        return bytes(out)

    def to_phase_bytes(self) -> bytes:
        """Mappa fase locale → luminosità per visualizzazione onde."""
        if self._is_torch():
            ph = self.phase.detach().cpu().numpy()
        else:
            ph = self.phase
        out = bytearray(self.width * self.height)
        for y in range(self.height):
            for x in range(self.width):
                v = (math.sin(float(ph[y, x])) + 1.0) * 0.5
                out[y * self.width + x] = max(0, min(255, int(v * 255)))
        return bytes(out)

    def stats(self) -> dict[str, Any]:
        reg = self.regional_energy()
        rec = self._last_recognition[:3]
        return {
            "width": self.width,
            "height": self.height,
            "pixels": self.neuron_count,
            "tick": self.tick,
            "backend": self._backend,
            "uses_gpu": self.uses_gpu,
            "temporal": True,
            "flux": round(self.flux_magnitude(), 5),
            "acceleration": round(self.acceleration_magnitude(), 5),
            "phase_coherence": round(self.phase_coherence(), 5),
            "recognized": [{"symbol": s, "score": round(sc, 4)} for s, sc in rec],
            "regions": {k: round(v, 5) for k, v in reg.items()},
            "mean_energy": round(sum(reg.values()) / max(1, len(reg)), 5),
            "mean_mass": round(self._mean_mass(), 5),
        }

    # --- physics ---

    def _step_torch(self) -> None:
        self.energy_t2 = self.energy_t1.clone()
        self.energy_t1 = self.energy.clone()

        vel = self.energy - self.energy_t1
        acc = vel - (self.energy_t1 - self.energy_t2)

        drive = self.potential + vel * INERTIA + acc * (INERTIA * 0.55)
        self.phase = (self.phase + self.omega * PHASE_DRIVE + drive * 0.25) % (2 * math.pi)

        wave = self._wavefront_torch(self.energy)
        act = self.energy.unsqueeze(0).unsqueeze(0)
        exc = F.conv2d(act, self._kernel_exc, padding=1).squeeze()
        inh = F.conv2d(act, self._kernel_inh, padding=1).squeeze()
        neighbor_phase = F.conv2d(
            self.phase.unsqueeze(0).unsqueeze(0),
            self._kernel_exc,
            padding=1,
        ).squeeze()
        gate = torch.cos(self.phase - neighbor_phase).clamp(0.0, 1.0)
        spread = (exc - inh * 0.4) * SYNAPSE_GAIN * gate

        gy, gx = torch.gradient(self.mass + self.energy * 0.35)
        self.vx = self.vx * 0.9 + gx * GRAVITY_G
        self.vy = self.vy * 0.9 + gy * GRAVITY_G

        new_e = self.energy * (1 - LEAK) + wave + spread + drive * 0.42
        new_e = self._advect_torch(new_e, self.vx, self.vy)
        self.energy = new_e.clamp(0, 1)
        self.potential = self.potential * (1 - LEAK * 1.2)

        persist = self.energy * self.energy_t1
        self.mass = torch.minimum(
            torch.tensor(3.0, device=self._torch_dev),
            self.mass + persist * PLASTICITY,
        )
        self.trace = torch.minimum(
            torch.tensor(1.0, device=self._torch_dev),
            self.trace * 0.96 + self.energy * 0.1 + acc.abs() * 0.08,
        )

    def _step_numpy(self) -> None:
        from organism.brain.retina_cortex import _convolve2d_numpy

        self.energy_t2 = self.energy_t1.copy()
        self.energy_t1 = self.energy.copy()

        vel = self.energy - self.energy_t1
        acc = vel - (self.energy_t1 - self.energy_t2)

        drive = self.potential + vel * INERTIA + acc * (INERTIA * 0.55)
        self.phase = (self.phase + self.omega * PHASE_DRIVE + drive * 0.25) % (2 * math.pi)

        wave = self._wavefront_numpy(self.energy)
        exc = _convolve2d_numpy(self.energy, self._kernel_exc)
        inh = _convolve2d_numpy(self.energy, self._kernel_inh)
        neighbor_phase = _convolve2d_numpy(self.phase, self._kernel_exc)
        gate = np.cos(self.phase - neighbor_phase).clip(0.0, 1.0)
        spread = (exc - inh * 0.4) * SYNAPSE_GAIN * gate

        gy, gx = np.gradient(self.mass + self.energy * 0.35)
        self.vx = self.vx * 0.9 + gx * GRAVITY_G
        self.vy = self.vy * 0.9 + gy * GRAVITY_G

        self.energy = np.clip(
            self.energy * (1 - LEAK) + wave + spread + drive * 0.42 + gx * 0.04 + gy * 0.04,
            0,
            1,
        )
        self.potential *= 1 - LEAK * 1.2

        persist = self.energy * self.energy_t1
        self.mass = np.minimum(3.0, self.mass + persist * PLASTICITY)
        self.trace = np.minimum(1.0, self.trace * 0.96 + self.energy * 0.1 + np.abs(acc) * 0.08)

    def _wavefront_torch(self, field: Any) -> Any:
        src = field.unsqueeze(0).unsqueeze(0)
        spread = F.conv2d(src, self._kernel_radial, padding=2).squeeze()
        lap = (
            F.conv2d(src, self._kernel_exc, padding=1).squeeze()
            - field * self._kernel_exc.sum()
        )
        return spread * WAVE_DECAY + lap * WAVE_SPEED * 0.15

    def _wavefront_numpy(self, field: Any) -> Any:
        from organism.brain.retina_cortex import _convolve2d_numpy

        spread = _convolve2d_numpy(field, self._kernel_radial)
        lap = _convolve2d_numpy(field, self._kernel_exc) - field * float(self._kernel_exc.sum())
        return spread * WAVE_DECAY + lap * WAVE_SPEED * 0.15

    def _velocity_field(self) -> Any:
        return self.energy - self.energy_t1

    def _acceleration_field(self) -> Any:
        vel = self._velocity_field()
        vel_t1 = self.energy_t1 - self.energy_t2
        return vel - vel_t1

    def _salience(self) -> Any:
        acc = self._acceleration_field()
        if self._is_torch():
            coh = torch.cos(self.phase)
            return self.energy * (0.45 + 0.35 * acc.abs()) + self.trace * 0.2 + coh.abs() * 0.08
        return self.energy * (0.45 + 0.35 * np.abs(acc)) + self.trace * 0.2 + np.abs(np.cos(self.phase)) * 0.08

    def _recognize_symbols(self, top_k: int = 4) -> list[tuple[str, float]]:
        y0, y1, x0, x1 = self._region_slice("associative")
        patch = self._energy_cpu_region(y0, y1, x0, x1)
        if not patch:
            return []
        scores: list[tuple[str, float]] = []
        for sym in SYMBOLS[:36]:
            tmpl = self._template_bank.get(sym)
            if not tmpl:
                continue
            sc = correlate_template(patch, tmpl)
            if sc > 0.12:
                scores.append((sym, sc))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]

    def _advect_torch(self, field: Any, vx: Any, vy: Any) -> Any:
        h, w = field.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=self._torch_dev),
            torch.linspace(-1, 1, w, device=self._torch_dev),
            indexing="ij",
        )
        scale_x = 2.0 / max(1, w)
        scale_y = 2.0 / max(1, h)
        grid = torch.stack((xx + vx * scale_x * 0.3, yy + vy * scale_y * 0.3), dim=-1).unsqueeze(0)
        out = F.grid_sample(
            field.unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return out.squeeze()

    def _init_omega_torch(self) -> Any:
        h, w = self.height, self.width
        rng = torch.Generator(device=self._torch_dev)
        rng.manual_seed(int(os.environ.get("ORGANISM_PHASE_SEED", "42")))
        noise = torch.rand((h, w), device=self._torch_dev, generator=rng) * 0.08
        base = torch.zeros((h, w), device=self._torch_dev)
        for name, bw in BAND_OMEGA.items():
            base += self._region_masks[name] * bw
        omega = base + noise
        k = torch.ones((1, 1, 5, 5), device=self._torch_dev) / 25.0
        return F.conv2d(omega.unsqueeze(0).unsqueeze(0), k, padding=2).squeeze()

    def _init_omega_numpy(self) -> Any:
        assert np is not None
        h, w = self.height, self.width
        rng = np.random.RandomState(int(os.environ.get("ORGANISM_PHASE_SEED", "42")))
        noise = rng.rand(h, w).astype(np.float32) * 0.08
        base = np.zeros((h, w), dtype=np.float32)
        for name, bw in BAND_OMEGA.items():
            base += self._region_masks[name] * bw
        omega = base + noise
        from organism.brain.retina_cortex import _convolve2d_numpy

        k = np.ones((5, 5), dtype=np.float32) / 25.0
        return _convolve2d_numpy(omega, k)

    def _radial_kernel_torch(self, dev: Any) -> Any:
        coords = torch.arange(-2, 3, device=dev, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        dist = torch.sqrt(xx * xx + yy * yy + 1e-6)
        k = torch.exp(-dist / RADIAL_SIGMA)
        return (k / k.sum()).view(1, 1, 5, 5)

    def _radial_kernel_numpy(self) -> Any:
        assert np is not None
        coords = np.arange(-2, 3, dtype=np.float32)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        dist = np.sqrt(xx * xx + yy * yy + 1e-6)
        k = np.exp(-dist / RADIAL_SIGMA)
        return (k / k.sum()).astype(np.float32)

    def _phase_kick_region(self, y0: int, y1: int, x0: int, x1: int, amount: float) -> None:
        if amount <= 0:
            return
        kick = amount * math.pi * 0.5
        if self._is_torch():
            self.phase[y0:y1, x0:x1] = (self.phase[y0:y1, x0:x1] + kick) % (2 * math.pi)
        else:
            self.phase[y0:y1, x0:x1] = (self.phase[y0:y1, x0:x1] + kick) % (2 * math.pi)

    def _inject_point(self, x: int, y: int, intensity: float, *, phase_kick: float = 0.0) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        if self._is_torch():
            self.potential[y, x] = min(1.0, float(self.potential[y, x].item()) + intensity)
            if phase_kick:
                self.phase[y, x] = float(self.phase[y, x].item() + phase_kick) % (2 * math.pi)
        else:
            self.potential[y, x] = min(1.0, float(self.potential[y, x]) + intensity)
            if phase_kick:
                self.phase[y, x] = float(self.phase[y, x] + phase_kick) % (2 * math.pi)

    def _mean_mass(self) -> float:
        if self._is_torch():
            return float(self.mass.mean().item())
        return float(self.mass.mean())

    def _add_masked(self, target: Any, mask: Any, intensity: float) -> None:
        if self._is_torch():
            target[:] = torch.minimum(
                torch.tensor(1.0, device=self._torch_dev),
                target + mask * intensity,
            )
        else:
            target[:] = np.minimum(1.0, target + mask * intensity)

    def _region_slice(self, region: str) -> tuple[int, int, int, int]:
        x0n, y0n, x1n, y1n = REGIONS[region]
        y0 = int(y0n * self.height)
        y1 = max(y0 + 1, int(y1n * self.height))
        x0 = int(x0n * self.width)
        x1 = max(x0 + 1, int(x1n * self.width))
        return y0, y1, x0, x1

    def _point_region(self, x: float, y: float) -> str:
        for name, (x0, y0, x1, y1) in REGIONS.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        best, best_score = "associative", 0.0
        for name, (x0, y0, x1, y1) in REGIONS.items():
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            d = (x - cx) ** 2 + (y - cy) ** 2
            score = 1.0 / (0.01 + d)
            if score > best_score:
                best_score, best = score, name
        return best

    def _mask_tensor(self, x0: float, y0: float, x1: float, y1: float) -> Any:
        h, w = self.height, self.width
        yy = torch.linspace(0, 1, h, device=self._torch_dev).view(h, 1)
        xx = torch.linspace(0, 1, w, device=self._torch_dev).view(1, w)
        return ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)).float()

    def _mask_numpy(self, x0: float, y0: float, x1: float, y1: float) -> Any:
        assert np is not None
        h, w = self.height, self.width
        yy = np.linspace(0, 1, h).reshape(h, 1)
        xx = np.linspace(0, 1, w).reshape(1, w)
        return ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)).astype(np.float32)

    def _normalize_grid(self, gray: list) -> list[list[float]]:
        if isinstance(gray, list) and gray and isinstance(gray[0], (int, float)):
            flat = [float(v) / 255.0 for v in gray]
            w, h = self.width, self.height
            return [flat[y * w : (y + 1) * w] for y in range(h) if y * w < len(flat)]
        return [[float(v) / 255.0 for v in row] for row in gray]

    def _energy_cpu(self) -> list[list[float]]:
        if self._is_torch():
            return self.energy.detach().cpu().tolist()
        return self.energy.tolist()

    def _energy_cpu_region(self, y0: int, y1: int, x0: int, x1: int) -> list[list[float]]:
        if self._is_torch():
            return self.energy[y0:y1, x0:x1].detach().cpu().tolist()
        return self.energy[y0:y1, x0:x1].tolist()

    def _is_torch(self) -> bool:
        return self._backend != "numpy" and HAS_TORCH and self._torch_dev is not None


def create_temporal_impulse_field(width: int, height: int, *, device: str = "auto") -> TemporalImpulseField:
    return TemporalImpulseField(width=width, height=height, device=device)
