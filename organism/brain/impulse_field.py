"""Mare di impulsi GPU — energia che si muove, non neuroni fissi.

L'impalcatura (codice) non decide: solo inietta sensoriale e legge.
La fisica del campo è separata dalla coscienza che lo osserva.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from organism.brain.gpu_backend import HAS_TORCH, resolve_device, scalar, torch_device

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore

if HAS_TORCH:
    import torch
    import torch.nn.functional as F

FIRE_THRESHOLD = 0.18
LEAK = 0.06
FLOW_GAIN = 0.55
SYNAPSE_GAIN = 0.38

# Regioni corticali (x0, y0, x1, y1) normalizzate 0-1
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "visual": (0.0, 0.0, 1.0, 0.32),
    "auditory": (0.0, 0.30, 0.22, 0.68),
    "associative": (0.20, 0.28, 0.80, 0.72),
    "memory": (0.78, 0.30, 1.0, 0.68),
    "motor": (0.0, 0.68, 1.0, 1.0),
}

_EXCITE = (
    (0.02, 0.07, 0.02),
    (0.07, 0.40, 0.07),
    (0.02, 0.07, 0.02),
)
_INHIBIT = (
    (0.05, 0.09, 0.05),
    (0.09, 0.0, 0.09),
    (0.05, 0.09, 0.05),
)


@dataclass
class ImpulseBlob:
    """Un impulso localizzato — energia in movimento."""

    x: float
    y: float
    energy: float
    vx: float
    vy: float
    region: str


@dataclass
class ImpulseField:
    """Campo 2D di impulsi — ogni pixel è energia libera di fluire."""

    width: int
    height: int
    device: str = "auto"
    energy: Any = field(init=False, repr=False)
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

    def __post_init__(self) -> None:
        if self.width < 16 or self.height < 16:
            raise ValueError("campo impulsi troppo piccolo")
        self._backend = resolve_device(self.device)
        h, w = self.height, self.width
        if self._backend != "numpy" and HAS_TORCH:
            if not HAS_TORCH:
                raise RuntimeError("torch richiesto per ImpulseField GPU")
            self._torch_dev = torch_device(self._backend)
            self.energy = torch.zeros((h, w), device=self._torch_dev, dtype=torch.float32)
            self.potential = torch.zeros((h, w), device=self._torch_dev, dtype=torch.float32)
            self.vx = torch.zeros((h, w), device=self._torch_dev, dtype=torch.float32)
            self.vy = torch.zeros((h, w), device=self._torch_dev, dtype=torch.float32)
            self.trace = torch.zeros((h, w), device=self._torch_dev, dtype=torch.float32)
            exc = torch.tensor(_EXCITE, device=self._torch_dev, dtype=torch.float32)
            inh = torch.tensor(_INHIBIT, device=self._torch_dev, dtype=torch.float32)
            self._kernel_exc = exc.view(1, 1, 3, 3)
            self._kernel_inh = inh.view(1, 1, 3, 3)
            self._region_masks = {
                name: self._mask_tensor(*box) for name, box in REGIONS.items()
            }
        elif HAS_NUMPY and np is not None:
            self._torch_dev = None
            self.energy = np.zeros((h, w), dtype=np.float32)
            self.potential = np.zeros((h, w), dtype=np.float32)
            self.vx = np.zeros((h, w), dtype=np.float32)
            self.vy = np.zeros((h, w), dtype=np.float32)
            self.trace = np.zeros((h, w), dtype=np.float32)
            self._kernel_exc = np.array(_EXCITE, dtype=np.float32)
            self._kernel_inh = np.array(_INHIBIT, dtype=np.float32)
            self._region_masks = {name: self._mask_numpy(*box) for name, box in REGIONS.items()}
        else:
            raise RuntimeError("ImpulseField richiede numpy o torch")

    @property
    def neuron_count(self) -> int:
        return self.width * self.height

    @property
    def uses_gpu(self) -> bool:
        return self._backend.startswith("cuda")

    def inject_region(
        self,
        region: str,
        intensity: float = 0.7,
        *,
        pattern: list[list[float]] | None = None,
    ) -> None:
        """Stimolo sensoriale — energia in una regione (impalcatura inietta, non comanda)."""
        mask = self._region_masks.get(region)
        if mask is None:
            return
        if pattern and self._is_torch():
            ph, pw = len(pattern), len(pattern[0]) if pattern else 0
            if ph > 0 and pw > 0:
                patch = torch.tensor(pattern, device=self._torch_dev, dtype=torch.float32)
                patch = patch / max(1e-6, patch.max())
                # resize patch into region bbox
                y0, y1, x0, x1 = self._region_slice(region)
                target = self.potential[y0:y1, x0:x1]
                th, tw = target.shape
                if th > 0 and tw > 0:
                    p4 = patch.unsqueeze(0).unsqueeze(0)
                    scaled = F.interpolate(p4, size=(th, tw), mode="bilinear", align_corners=False).squeeze()
                    self.potential[y0:y1, x0:x1] = torch.minimum(
                        torch.tensor(1.0, device=self._torch_dev),
                        target + scaled * intensity * mask[y0:y1, x0:x1],
                    )
                    return
        self._add_masked(self.potential, mask, intensity)

    def inject_pixels(self, gray: list[list[float]] | list[list[int]] | list[int], *, gain: float = 0.9) -> None:
        """Webcam → regione visiva."""
        grid = self._normalize_grid(gray)
        if not grid:
            return
        gh, gw = len(grid), len(grid[0]) if grid else 0
        if self._is_torch():
            patch = torch.tensor(grid, device=self._torch_dev, dtype=torch.float32)
            y0, y1, x0, x1 = self._region_slice("visual")
            th, tw = y1 - y0, x1 - x0
            if th > 0 and tw > 0:
                p4 = patch.unsqueeze(0).unsqueeze(0)
                scaled = F.interpolate(p4, size=(th, tw), mode="bilinear", align_corners=False).squeeze()
                m = self._region_masks["visual"][y0:y1, x0:x1]
                self.potential[y0:y1, x0:x1] = torch.minimum(
                    torch.tensor(1.0, device=self._torch_dev),
                    self.potential[y0:y1, x0:x1] + scaled * gain * m,
                )
        elif HAS_NUMPY and np is not None:
            patch = np.array([row[:gw] for row in grid[:gh]], dtype=np.float32)
            y0, y1, x0, x1 = self._region_slice("visual")
            target_h, target_w = y1 - y0, x1 - x0
            if target_h > 0 and target_w > 0:
                ys = np.linspace(0, max(0, gh - 1), target_h).astype(int)
                xs = np.linspace(0, max(0, gw - 1), target_w).astype(int)
                sampled = patch[ys][:, xs]
                m = self._region_masks["visual"][y0:y1, x0:x1]
                self.potential[y0:y1, x0:x1] = np.minimum(
                    1.0, self.potential[y0:y1, x0:x1] + sampled * gain * m
                )

    def inject_audio_band(self, bands: list[float], *, gain: float = 0.5) -> None:
        """Bande frequenza → regione uditiva."""
        if not bands:
            return
        for i, b in enumerate(bands[:8]):
            row = i % 4
            col = i // 4
            x = int(self.width * (0.05 + col * 0.08))
            y = int(self.height * (0.35 + row * 0.08))
            self._inject_point(x, y, float(b) * gain)

    def inject_text_energy(self, dim_energies: list[float], *, gain: float = 0.45) -> None:
        """Hash semantico → regione associativa."""
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
            t4 = t.unsqueeze(0).unsqueeze(0)
            scaled = F.interpolate(t4, size=(th, tw), mode="bilinear", align_corners=False).squeeze()
            self.potential[y0:y1, x0:x1] = torch.minimum(
                torch.tensor(1.0, device=self._torch_dev),
                self.potential[y0:y1, x0:x1] + scaled * gain,
            )
        elif HAS_NUMPY and np is not None:
            patch = np.array(padded, dtype=np.float32).reshape(side, side)
            patch = patch / max(1e-6, float(patch.max()))
            ys = np.linspace(0, side - 1, th).astype(int)
            xs = np.linspace(0, side - 1, tw).astype(int)
            scaled = patch[ys][:, xs]
            self.potential[y0:y1, x0:x1] = np.minimum(
                1.0, self.potential[y0:y1, x0:x1] + scaled * gain
            )

    def inject_memory_echo(self, signature: list[float], *, gain: float = 0.25) -> None:
        """Richiamo mnemonico — debole reiniezione in regione memoria."""
        if not signature:
            return
        y0, y1, x0, x1 = self._region_slice("memory")
        n = min(len(signature), (y1 - y0) * (x1 - x0))
        if self._is_torch() and n > 0:
            side = int(n**0.5)
            t = torch.tensor(signature[: side * side], device=self._torch_dev, dtype=torch.float32).view(side, side)
            t4 = t.unsqueeze(0).unsqueeze(0)
            th, tw = y1 - y0, x1 - x0
            scaled = F.interpolate(t4, size=(th, tw), mode="bilinear", align_corners=False).squeeze()
            self.potential[y0:y1, x0:x1] += scaled * gain

    def step(self, steps: int = 1) -> None:
        """Un tick fisico — impulsi si muovono e si collegano (sinapsi virtuali)."""
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
        """Quanto si stanno muovendo gli impulsi."""
        if self._is_torch():
            return float(torch.sqrt(self.vx * self.vx + self.vy * self.vy).mean().item())
        return float(np.sqrt(self.vx * self.vx + self.vy * self.vy).mean())

    def active_blobs(self, k: int = 12) -> list[ImpulseBlob]:
        """Impulsi dominanti — coordinate precise per la coscienza."""
        sal = self._salience()
        blobs: list[ImpulseBlob] = []
        if self._is_torch():
            flat = sal.reshape(-1)
            vals, idxs = torch.topk(flat, min(k * 6, flat.numel()))
            taken = torch.zeros(sal.shape, dtype=torch.bool, device=self._torch_dev)
            for idx_t, val_t in zip(idxs.tolist(), vals.tolist()):
                if len(blobs) >= k or val_t < 0.04:
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
                if v < 0.04:
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
        """Impronta compressa del campo — per memoria episodica."""
        if self._is_torch():
            e = self.energy.detach().cpu().flatten()
            step = max(1, e.numel() // 128)
            return [float(v) for v in e[::step][:128]]
        flat = self.energy.ravel()
        step = max(1, len(flat) // 128)
        return [float(v) for v in flat[::step][:128]]

    def to_energy_bytes(self) -> bytes:
        grid = self._energy_cpu()
        out = bytearray(self.width * self.height)
        for y, row in enumerate(grid):
            for x, v in enumerate(row):
                out[y * self.width + x] = max(0, min(255, int(v * 255)))
        return bytes(out)

    def stats(self) -> dict[str, Any]:
        reg = self.regional_energy()
        return {
            "width": self.width,
            "height": self.height,
            "pixels": self.neuron_count,
            "tick": self.tick,
            "backend": self._backend,
            "uses_gpu": self.uses_gpu,
            "flux": round(self.flux_magnitude(), 5),
            "regions": {k: round(v, 5) for k, v in reg.items()},
            "mean_energy": round(sum(reg.values()) / max(1, len(reg)), 5),
        }

    # --- physics internals ---

    def _step_torch(self) -> None:
        e = self.energy
        p = self.potential
        fired = p >= FIRE_THRESHOLD
        self.energy = torch.where(fired, torch.minimum(torch.tensor(1.0, device=self._torch_dev), e + p * 0.45), e * (1 - LEAK))
        self.potential = torch.where(fired, p * 0.25, p * (1 - LEAK))

        act = self.energy.unsqueeze(0).unsqueeze(0)
        exc = F.conv2d(act, self._kernel_exc, padding=1).squeeze()
        inh = F.conv2d(act, self._kernel_inh, padding=1).squeeze()
        spread = (exc - inh * 0.42) * SYNAPSE_GAIN
        self.potential = torch.minimum(torch.tensor(1.0, device=self._torch_dev), self.potential + spread)

        gy, gx = torch.gradient(self.energy)
        self.vx = self.vx * 0.88 + gx * FLOW_GAIN
        self.vy = self.vy * 0.88 + gy * FLOW_GAIN
        self.energy = self._advect_torch(self.energy, self.vx, self.vy)
        self.trace = torch.minimum(torch.tensor(1.0, device=self._torch_dev), self.trace * 0.97 + self.energy * 0.08)

    def _step_numpy(self) -> None:
        fired = self.potential >= FIRE_THRESHOLD
        self.energy = np.where(fired, np.minimum(1.0, self.energy + self.potential * 0.45), self.energy * (1 - LEAK))
        self.potential = np.where(fired, self.potential * 0.25, self.potential * (1 - LEAK))
        from organism.brain.retina_cortex import _convolve2d_numpy

        exc = _convolve2d_numpy(self.energy, self._kernel_exc)
        inh = _convolve2d_numpy(self.energy, self._kernel_inh)
        self.potential = np.minimum(1.0, self.potential + (exc - inh * 0.42) * SYNAPSE_GAIN)
        gy, gx = np.gradient(self.energy)
        self.vx = self.vx * 0.88 + gx * FLOW_GAIN
        self.vy = self.vy * 0.88 + gy * FLOW_GAIN
        self.energy = np.clip(self.energy + gx * 0.05 + gy * 0.05, 0, 1)
        self.trace = np.minimum(1.0, self.trace * 0.97 + self.energy * 0.08)

    def _advect_torch(self, field: Any, vx: Any, vy: Any) -> Any:
        h, w = field.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=self._torch_dev),
            torch.linspace(-1, 1, w, device=self._torch_dev),
            indexing="ij",
        )
        scale_x = 2.0 / max(1, w)
        scale_y = 2.0 / max(1, h)
        grid_x = xx + vx * scale_x * 0.35
        grid_y = yy + vy * scale_y * 0.35
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        src = field.unsqueeze(0).unsqueeze(0)
        out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out.squeeze().clamp(0, 1)

    def _salience(self) -> Any:
        if self._is_torch():
            gy, gx = torch.gradient(self.energy)
            grad = torch.sqrt(gx * gx + gy * gy)
            return self.energy * (0.5 + 0.5 * grad) + self.trace * 0.15
        gy, gx = np.gradient(self.energy)
        grad = np.sqrt(gx * gx + gy * gy)
        return self.energy * (0.5 + 0.5 * grad) + self.trace * 0.15

    def _inject_point(self, x: int, y: int, intensity: float) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        if self._is_torch():
            self.potential[y, x] = min(1.0, float(self.potential[y, x].item()) + intensity)
        else:
            self.potential[y, x] = min(1.0, float(self.potential[y, x]) + intensity)

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
        best = "associative"
        best_score = 0.0
        for name, (x0, y0, x1, y1) in REGIONS.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            d = (x - cx) ** 2 + (y - cy) ** 2
            score = 1.0 / (0.01 + d)
            if score > best_score:
                best_score = score
                best = name
        return best

    def _mask_tensor(self, x0: float, y0: float, x1: float, y1: float) -> Any:
        h, w = self.height, self.width
        yy = torch.linspace(0, 1, h, device=self._torch_dev).view(h, 1)
        xx = torch.linspace(0, 1, w, device=self._torch_dev).view(1, w)
        inside = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        return inside.float()

    def _mask_numpy(self, x0: float, y0: float, x1: float, y1: float) -> Any:
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

    def _is_torch(self) -> bool:
        return self._backend != "numpy" and HAS_TORCH and self._torch_dev is not None


def create_impulse_field(width: int, height: int, *, device: str = "auto") -> ImpulseField:
    import os

    if os.environ.get("ORGANISM_TEMPORAL", "1") != "0":
        from organism.brain.temporal_impulse_field import create_temporal_impulse_field

        return create_temporal_impulse_field(width, height, device=device)  # type: ignore[return-value]
    return ImpulseField(width=width, height=height, device=device)
