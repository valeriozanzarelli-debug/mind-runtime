"""Campo impulsi 3D — ogni voxel è un neurone-pixel con sinapsi virtuali conv3d.

Con la stessa VRAM del campo 2D si ottiene W×H×D voxel: es. 512×384×128 ≈ 25M neuroni
vs 4096×3072 ≈ 12.6M in 2D piatto — quasi il doppio, con topologia spaziale profonda.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from organism.brain.gpu_backend import HAS_TORCH, resolve_device, torch_device
from organism.brain.impulse_field import (
    FIRE_THRESHOLD,
    FLOW_GAIN,
    LEAK,
    SYNAPSE_GAIN,
    ImpulseBlob,
    _EXCITE,
    _INHIBIT,
)

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False

if HAS_TORCH:
    import torch
    import torch.nn.functional as F

# Regioni corticali 3D (x0,y0,z0,x1,y1,z1) normalizzate 0-1
REGIONS_3D: dict[str, tuple[float, float, float, float, float, float]] = {
    "visual": (0.0, 0.0, 0.0, 1.0, 0.35, 0.28),
    "auditory": (0.0, 0.28, 0.15, 0.22, 0.72, 0.55),
    "associative": (0.18, 0.22, 0.20, 0.82, 0.78, 0.80),
    "memory": (0.76, 0.25, 0.55, 1.0, 0.72, 1.0),
    "motor": (0.0, 0.72, 0.0, 1.0, 1.0, 0.35),
}

_EXCITE_3D = tuple(tuple(row for row in plane) for plane in (
    ((0.01, 0.04, 0.01), (0.04, 0.12, 0.04), (0.01, 0.04, 0.01)),
    ((0.04, 0.12, 0.04), (0.12, 0.35, 0.12), (0.04, 0.12, 0.04)),
    ((0.01, 0.04, 0.01), (0.04, 0.12, 0.04), (0.01, 0.04, 0.01)),
))
_INHIBIT_3D = tuple(tuple(row for row in plane) for plane in (
    ((0.03, 0.06, 0.03), (0.06, 0.0, 0.06), (0.03, 0.06, 0.03)),
    ((0.06, 0.0, 0.06), (0.0, 0.0, 0.0), (0.06, 0.0, 0.06)),
    ((0.03, 0.06, 0.03), (0.06, 0.0, 0.06), (0.03, 0.06, 0.03)),
))


@dataclass
class ImpulseField3D:
    """Volume 3D di impulsi — shape (depth, height, width) = (D, H, W)."""

    width: int
    height: int
    depth: int
    device: str = "auto"
    energy: Any = field(init=False, repr=False)
    potential: Any = field(init=False, repr=False)
    vx: Any = field(init=False, repr=False)
    vy: Any = field(init=False, repr=False)
    vz: Any = field(init=False, repr=False)
    trace: Any = field(init=False, repr=False)
    tick: float = 0.0
    _backend: str = field(init=False, repr=False)
    _torch_dev: Any = field(init=False, repr=False, default=None)
    _region_masks: dict[str, Any] = field(init=False, repr=False)
    _kernel_exc: Any = field(init=False, repr=False)
    _kernel_inh: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.depth) < 8:
            raise ValueError("campo 3D troppo piccolo (min 8 per asse)")
        self._backend = resolve_device(self.device)
        d, h, w = self.depth, self.height, self.width
        if self._backend != "numpy" and HAS_TORCH:
            dev = torch_device(self._backend)
            self._torch_dev = dev
            shape = (d, h, w)
            self.energy = torch.zeros(shape, device=dev, dtype=torch.float32)
            self.potential = torch.zeros(shape, device=dev, dtype=torch.float32)
            self.vx = torch.zeros(shape, device=dev, dtype=torch.float32)
            self.vy = torch.zeros(shape, device=dev, dtype=torch.float32)
            self.vz = torch.zeros(shape, device=dev, dtype=torch.float32)
            self.trace = torch.zeros(shape, device=dev, dtype=torch.float32)
            exc = torch.tensor(_EXCITE_3D, device=dev, dtype=torch.float32)
            inh = torch.tensor(_INHIBIT_3D, device=dev, dtype=torch.float32)
            self._kernel_exc = exc.view(1, 1, 3, 3, 3)
            self._kernel_inh = inh.view(1, 1, 3, 3, 3)
            self._region_masks = {n: self._mask_tensor(*box) for n, box in REGIONS_3D.items()}
        elif HAS_NUMPY and np is not None:
            self._torch_dev = None
            shape = (d, h, w)
            self.energy = np.zeros(shape, dtype=np.float32)
            self.potential = np.zeros(shape, dtype=np.float32)
            self.vx = np.zeros(shape, dtype=np.float32)
            self.vy = np.zeros(shape, dtype=np.float32)
            self.vz = np.zeros(shape, dtype=np.float32)
            self.trace = np.zeros(shape, dtype=np.float32)
            self._kernel_exc = np.array(_EXCITE_3D, dtype=np.float32)
            self._kernel_inh = np.array(_INHIBIT_3D, dtype=np.float32)
            self._region_masks = {n: self._mask_numpy(*box) for n, box in REGIONS_3D.items()}
        else:
            raise RuntimeError("ImpulseField3D richiede numpy o torch")

    @property
    def neuron_count(self) -> int:
        return self.width * self.height * self.depth

    @property
    def uses_gpu(self) -> bool:
        return self._backend != "numpy"

    @property
    def dimensions(self) -> str:
        return f"{self.width}x{self.height}x{self.depth}"

    def inject_pixels(self, gray: list, *, gain: float = 0.9) -> None:
        """Visione → slice frontale z=0 (corteccia visiva anteriore)."""
        grid = self._normalize_grid(gray)
        z = 0
        for y, row in enumerate(grid[: self.height]):
            for x, val in enumerate(row[: self.width]):
                self._inject_voxel(z, y, x, float(val) * gain)

    def inject_audio_band(self, bands: list[float]) -> None:
        if not bands:
            return
        mask = self._region_masks.get("auditory")
        if mask is None:
            return
        scaled = sum(bands[:16]) / max(1, len(bands[:16]))
        self._add_masked(self.potential, mask, scaled * 0.55)

    def inject_text_energy(self, energies: list[float]) -> None:
        if not energies:
            return
        mask = self._region_masks.get("associative")
        if mask is None:
            return
        intensity = sum(energies[:32]) / max(1, len(energies[:32]))
        self._add_masked(self.potential, mask, intensity * 0.45)

    def inject_memory_echo(self, signature: list[float], *, gain: float = 0.12) -> None:
        if not signature:
            return
        mask = self._region_masks.get("memory")
        if mask is None:
            return
        intensity = min(1.0, sum(abs(v) for v in signature[:64]) / max(1, len(signature[:64])))
        self._add_masked(self.potential, mask, intensity * gain)

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self.tick += 1.0
            if self._is_torch():
                self._step_torch()
            else:
                self._step_numpy()

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
        if self._is_torch():
            v = self.vx * self.vx + self.vy * self.vy + self.vz * self.vz
            return float(torch.sqrt(v).mean().item())
        return float(np.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz).mean())

    def active_blobs(self, k: int = 12) -> list[ImpulseBlob]:
        sal = self._salience()
        blobs: list[ImpulseBlob] = []
        if self._is_torch():
            flat = sal.reshape(-1)
            vals, idxs = torch.topk(flat, min(k * 8, flat.numel()))
            taken = torch.zeros(sal.shape, dtype=torch.bool, device=self._torch_dev)
            hw = self.width
            hh = self.height
            for idx_t, val_t in zip(idxs.tolist(), vals.tolist()):
                if len(blobs) >= k or val_t < 0.04:
                    break
                flat_i = int(idx_t)
                z, rem = divmod(flat_i, hh * hw)
                y, x = divmod(rem, hw)
                if taken[z, y, x]:
                    continue
                px, py, pz = x / hw, y / hh, z / self.depth
                blobs.append(
                    ImpulseBlob(
                        x=px,
                        y=py,
                        energy=float(val_t),
                        vx=float(self.vx[z, y, x].item()),
                        vy=float(self.vy[z, y, x].item()),
                        region=self._point_region(px, py, pz),
                        z=pz,
                    )
                )
                z0, z1 = max(0, z - 2), min(self.depth, z + 3)
                y0, y1 = max(0, y - 4), min(self.height, y + 5)
                x0, x1 = max(0, x - 4), min(self.width, x + 5)
                taken[z0:z1, y0:y1, x0:x1] = True
        else:
            order = np.argsort(sal.ravel())[::-1]
            taken = np.zeros(sal.shape, dtype=bool)
            hw, hh = self.width, self.height
            for flat_i in order:
                if len(blobs) >= k:
                    break
                flat_i = int(flat_i)
                z, rem = divmod(flat_i, hh * hw)
                y, x = divmod(rem, hw)
                v = float(sal[z, y, x])
                if v < 0.04:
                    break
                if taken[z, y, x]:
                    continue
                px, py, pz = x / hw, y / hh, z / self.depth
                blobs.append(
                    ImpulseBlob(
                        x=px,
                        y=py,
                        energy=v,
                        vx=float(self.vx[z, y, x]),
                        vy=float(self.vy[z, y, x]),
                        region=self._point_region(px, py, pz),
                        z=pz,
                    )
                )
                taken[max(0, z - 2) : z + 3, max(0, y - 4) : y + 5, max(0, x - 4) : x + 5] = True
        return blobs

    def signature(self) -> list[float]:
        if self._is_torch():
            e = self.energy.detach().cpu().flatten()
            step = max(1, e.numel() // 128)
            return [float(v) for v in e[::step][:128]]
        flat = self.energy.ravel()
        step = max(1, len(flat) // 128)
        return [float(v) for v in flat[::step][:128]]

    def to_energy_bytes(self) -> bytes:
        """Proiezione max-depth → immagine 2D per UI."""
        if self._is_torch():
            proj = self.energy.max(dim=0).values.detach().cpu()
            grid = proj.tolist()
        else:
            grid = np.max(self.energy, axis=0).tolist()
        out = bytearray(self.width * self.height)
        for y, row in enumerate(grid[: self.height]):
            for x, v in enumerate(row[: self.width]):
                out[y * self.width + x] = max(0, min(255, int(float(v) * 255)))
        return bytes(out)

    def stats(self) -> dict[str, Any]:
        reg = self.regional_energy()
        return {
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "dimensions": self.dimensions,
            "pixels": self.neuron_count,
            "voxels": self.neuron_count,
            "tick": self.tick,
            "backend": self._backend,
            "uses_gpu": self.uses_gpu,
            "spatial": "3d",
            "flux": round(self.flux_magnitude(), 5),
            "regions": {k: round(v, 5) for k, v in reg.items()},
            "mean_energy": round(sum(reg.values()) / max(1, len(reg)), 5),
        }

    def _step_torch(self) -> None:
        e, p = self.energy, self.potential
        fired = p >= FIRE_THRESHOLD
        one = torch.tensor(1.0, device=self._torch_dev)
        self.energy = torch.where(fired, torch.minimum(one, e + p * 0.45), e * (1 - LEAK))
        self.potential = torch.where(fired, p * 0.25, p * (1 - LEAK))

        act = self.energy.unsqueeze(0).unsqueeze(0)
        exc = F.conv3d(act, self._kernel_exc, padding=1).squeeze()
        inh = F.conv3d(act, self._kernel_inh, padding=1).squeeze()
        self.potential = torch.minimum(one, self.potential + (exc - inh * 0.42) * SYNAPSE_GAIN)

        gz, gy, gx = torch.gradient(self.energy)
        self.vx = self.vx * 0.88 + gx * FLOW_GAIN
        self.vy = self.vy * 0.88 + gy * FLOW_GAIN
        self.vz = self.vz * 0.88 + gz * FLOW_GAIN * 0.85
        self.energy = torch.clamp(self.energy + gx * 0.04 + gy * 0.04 + gz * 0.03, 0, 1)
        self.trace = torch.minimum(one, self.trace * 0.97 + self.energy * 0.08)

    def _step_numpy(self) -> None:
        fired = self.potential >= FIRE_THRESHOLD
        self.energy = np.where(fired, np.minimum(1.0, self.energy + self.potential * 0.45), self.energy * (1 - LEAK))
        self.potential = np.where(fired, self.potential * 0.25, self.potential * (1 - LEAK))
        exc = _convolve3d_numpy(self.energy, self._kernel_exc)
        inh = _convolve3d_numpy(self.energy, self._kernel_inh)
        self.potential = np.minimum(1.0, self.potential + (exc - inh * 0.42) * SYNAPSE_GAIN)
        gz, gy, gx = np.gradient(self.energy)
        self.vx = self.vx * 0.88 + gx * FLOW_GAIN
        self.vy = self.vy * 0.88 + gy * FLOW_GAIN
        self.vz = self.vz * 0.88 + gz * FLOW_GAIN * 0.85
        self.energy = np.clip(self.energy + gx * 0.04 + gy * 0.04 + gz * 0.03, 0, 1)
        self.trace = np.minimum(1.0, self.trace * 0.97 + self.energy * 0.08)

    def _salience(self) -> Any:
        if self._is_torch():
            gz, gy, gx = torch.gradient(self.energy)
            grad = torch.sqrt(gx * gx + gy * gy + gz * gz)
            return self.energy * (0.5 + 0.5 * grad) + self.trace * 0.15
        gz, gy, gx = np.gradient(self.energy)
        grad = np.sqrt(gx * gx + gy * gy + gz * gz)
        return self.energy * (0.5 + 0.5 * grad) + self.trace * 0.15

    def _inject_voxel(self, z: int, y: int, x: int, intensity: float) -> None:
        if not (0 <= z < self.depth and 0 <= y < self.height and 0 <= x < self.width):
            return
        if self._is_torch():
            self.potential[z, y, x] = min(1.0, float(self.potential[z, y, x].item()) + intensity)
        else:
            self.potential[z, y, x] = min(1.0, float(self.potential[z, y, x]) + intensity)

    def _add_masked(self, target: Any, mask: Any, intensity: float) -> None:
        if self._is_torch():
            target[:] = torch.minimum(
                torch.tensor(1.0, device=self._torch_dev),
                target + mask * intensity,
            )
        else:
            target[:] = np.minimum(1.0, target + mask * intensity)

    def _point_region(self, x: float, y: float, z: float) -> str:
        for name, (x0, y0, z0, x1, y1, z1) in REGIONS_3D.items():
            if x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1:
                return name
        return "associative"

    def _mask_tensor(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> Any:
        d, h, w = self.depth, self.height, self.width
        zz = torch.linspace(0, 1, d, device=self._torch_dev).view(d, 1, 1)
        yy = torch.linspace(0, 1, h, device=self._torch_dev).view(1, h, 1)
        xx = torch.linspace(0, 1, w, device=self._torch_dev).view(1, 1, w)
        inside = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1) & (zz >= z0) & (zz <= z1)
        return inside.float()

    def _mask_numpy(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> Any:
        d, h, w = self.depth, self.height, self.width
        zz = np.linspace(0, 1, d).reshape(d, 1, 1)
        yy = np.linspace(0, 1, h).reshape(1, h, 1)
        xx = np.linspace(0, 1, w).reshape(1, 1, w)
        return ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1) & (zz >= z0) & (zz <= z1)).astype(np.float32)

    def _normalize_grid(self, gray: list) -> list[list[float]]:
        if isinstance(gray, list) and gray and isinstance(gray[0], (int, float)):
            flat = [float(v) / 255.0 for v in gray]
            w, h = self.width, self.height
            return [flat[y * w : (y + 1) * w] for y in range(h) if y * w < len(flat)]
        return [[float(v) / 255.0 for v in row] for row in gray]

    def _is_torch(self) -> bool:
        return self._backend != "numpy" and HAS_TORCH and self._torch_dev is not None


def _convolve3d_numpy(vol: Any, kernel: Any) -> Any:
    """Conv3d valid padding=1 — sufficiente per test CPU."""
    kd, kh, kw = kernel.shape
    pd, ph, pw = kd // 2, kh // 2, kw // 2
    d, h, w = vol.shape
    out = np.zeros_like(vol)
    padded = np.pad(vol, ((pd, pd), (ph, ph), (pw, pw)), mode="edge")
    for z in range(d):
        for y in range(h):
            for x in range(w):
                patch = padded[z : z + kd, y : y + kh, x : x + kw]
                out[z, y, x] = float((patch * kernel).sum())
    return out


def create_impulse_field_3d(
    width: int,
    height: int,
    depth: int,
    *,
    device: str = "auto",
) -> ImpulseField3D:
    return ImpulseField3D(width=width, height=height, depth=depth, device=device)
