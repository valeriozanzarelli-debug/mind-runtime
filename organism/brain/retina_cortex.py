"""Retina cortex — ogni pixel è un neurone, sinapsi virtuali via kernel locale.

Backend:
- ``numpy`` — CPU puro (default senza torch)
- ``cpu`` / ``cuda`` — PyTorch (GPU su Windows con driver NVIDIA + CUDA)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from organism.brain.gpu_backend import (
    HAS_TORCH,
    cuda_available,
    gpu_info,
    resolve_device,
    scalar,
    torch_device,
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

# Kernel messicano: eccitazione centrale + inibizione periferica (corteccia V1-like)
_EXCITE = (
    (0.02, 0.06, 0.02),
    (0.06, 0.35, 0.06),
    (0.02, 0.06, 0.02),
)
_INHIBIT = (
    (0.04, 0.08, 0.04),
    (0.08, 0.0, 0.08),
    (0.04, 0.08, 0.04),
)

FIRE_THRESHOLD = 0.22
LEAK_RATE = 0.08


@dataclass
class LongRangeSynapse:
    """Sinapsi virtuale sparsa tra due coordinate pixel-neurone."""

    sy: int
    sx: int
    dy: int
    dx: int
    weight: float = 0.15


@dataclass
class RetinaCortex:
    """Campo neurale 2D — width × height neuroni, uno per pixel."""

    width: int
    height: int
    device: str = "auto"
    activation: Any = field(init=False, repr=False)
    potential: Any = field(init=False, repr=False)
    tick: float = 0.0
    long_range: list[LongRangeSynapse] = field(default_factory=list)
    _kernel_exc: Any = field(init=False, repr=False)
    _kernel_inh: Any = field(init=False, repr=False)
    _backend: str = field(init=False, repr=False)
    _torch_dev: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.width < 4 or self.height < 4:
            raise ValueError("retina troppo piccola (min 4×4)")
        self._backend = resolve_device(self.device)
        if self._backend in ("cuda", "cpu") or self._backend.startswith("cuda"):
            if not HAS_TORCH:
                raise RuntimeError(
                    "torch richiesto per GPU. Su Windows:\n"
                    "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                    "  pip install -e \".[gpu]\""
                )
            self._torch_dev = torch_device(self._backend)
            self.activation = torch.zeros(
                (self.height, self.width), device=self._torch_dev, dtype=torch.float32
            )
            self.potential = torch.zeros(
                (self.height, self.width), device=self._torch_dev, dtype=torch.float32
            )
            exc = torch.tensor(_EXCITE, device=self._torch_dev, dtype=torch.float32)
            inh = torch.tensor(_INHIBIT, device=self._torch_dev, dtype=torch.float32)
            self._kernel_exc = exc.view(1, 1, 3, 3)
            self._kernel_inh = inh.view(1, 1, 3, 3)
            self._fire_threshold = torch.tensor(FIRE_THRESHOLD, device=self._torch_dev)
        elif HAS_NUMPY and np is not None:
            self.activation = np.zeros((self.height, self.width), dtype=np.float32)
            self.potential = np.zeros((self.height, self.width), dtype=np.float32)
            self._kernel_exc = np.array(_EXCITE, dtype=np.float32)
            self._kernel_inh = np.array(_INHIBIT, dtype=np.float32)
        else:
            self.activation = [[0.0] * self.width for _ in range(self.height)]
            self.potential = [[0.0] * self.width for _ in range(self.height)]
            self._kernel_exc = _EXCITE
            self._kernel_inh = _INHIBIT

    @property
    def neuron_count(self) -> int:
        return self.width * self.height

    @property
    def uses_gpu(self) -> bool:
        return self._backend.startswith("cuda")

    def inject_pixels(
        self,
        gray: list[list[float]] | list[list[int]] | list[int],
        *,
        width: int | None = None,
        height: int | None = None,
        gain: float = 0.85,
    ) -> int:
        """Stimolo sensoriale diretto — luminanza pixel → potenziale neurale."""
        if isinstance(gray, list) and gray and isinstance(gray[0], (int, float)):
            flat = [float(v) for v in gray]  # type: ignore[arg-type]
            w = width or self.width
            h = height or self.height
            grid = []
            for y in range(h):
                row = []
                for x in range(w):
                    i = y * w + x
                    row.append(flat[i] / 255.0 if i < len(flat) else 0.0)
                grid.append(row)
        else:
            grid = [[float(v) / 255.0 for v in row] for row in gray]  # type: ignore[union-attr]

        gh = min(len(grid), self.height)
        gw = min(len(grid[0]) if grid else 0, self.width)

        if self._is_torch():
            patch = torch.tensor(
                [row[:gw] for row in grid[:gh]],
                device=self._torch_dev,
                dtype=torch.float32,
            )
            self.potential[:gh, :gw] = torch.minimum(
                torch.tensor(1.0, device=self._torch_dev),
                self.potential[:gh, :gw] + patch * gain,
            )
            mask = self.potential[:gh, :gw] >= FIRE_THRESHOLD
            self.activation[:gh, :gw] = torch.where(
                mask,
                torch.minimum(
                    torch.tensor(1.0, device=self._torch_dev),
                    self.activation[:gh, :gw] + self.potential[:gh, :gw] * 0.5,
                ),
                self.activation[:gh, :gw],
            )
            return int(mask.sum().item())

        if HAS_NUMPY and np is not None and not isinstance(self.activation, list):
            patch = np.array([row[:gw] for row in grid[:gh]], dtype=np.float32)
            self.potential[:gh, :gw] = np.minimum(
                1.0, self.potential[:gh, :gw] + patch * gain
            )
            mask = self.potential[:gh, :gw] >= FIRE_THRESHOLD
            self.activation[:gh, :gw] = np.where(
                mask,
                np.minimum(1.0, self.activation[:gh, :gw] + self.potential[:gh, :gw] * 0.5),
                self.activation[:gh, :gw],
            )
            return int(mask.sum())

        fired = 0
        for y in range(gh):
            for x in range(gw):
                self.potential[y][x] = min(1.0, self.potential[y][x] + grid[y][x] * gain)
                if self.potential[y][x] >= FIRE_THRESHOLD:
                    self.activation[y][x] = min(
                        1.0, self.activation[y][x] + self.potential[y][x] * 0.5
                    )
                    fired += 1
        return fired

    def inject_point(self, x: int, y: int, intensity: float = 0.9) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        if self._is_torch():
            self.potential[y, x] = min(1.0, float(self.potential[y, x].item()) + intensity)
            if self.potential[y, x] >= FIRE_THRESHOLD:
                self.activation[y, x] = min(
                    1.0, float(self.activation[y, x].item()) + intensity * 0.6
                )
            return
        if HAS_NUMPY and np is not None and not isinstance(self.activation, list):
            self.potential[y, x] = min(1.0, float(self.potential[y, x]) + intensity)
            if self.potential[y, x] >= FIRE_THRESHOLD:
                self.activation[y, x] = min(1.0, float(self.activation[y, x]) + intensity * 0.6)
        else:
            self.potential[y][x] = min(1.0, self.potential[y][x] + intensity)
            if self.potential[y][x] >= FIRE_THRESHOLD:
                self.activation[y][x] = min(1.0, self.activation[y][x] + intensity * 0.6)

    def propagate(self, steps: int = 1, *, decay: float = LEAK_RATE) -> None:
        for _ in range(steps):
            self.tick += 1.0
            if self._is_torch():
                self._propagate_torch(decay)
            elif HAS_NUMPY and np is not None and not isinstance(self.activation, list):
                self._propagate_numpy(decay)
            else:
                self._propagate_python(decay)
            for syn in self.long_range:
                self._apply_long_range(syn)

    def _propagate_torch(self, decay: float) -> None:
        act = self.activation.unsqueeze(0).unsqueeze(0)
        exc = F.conv2d(act, self._kernel_exc, padding=1).squeeze()
        inh = F.conv2d(act, self._kernel_inh, padding=1).squeeze()
        delta = exc - inh * 0.45
        self.potential = torch.minimum(
            torch.tensor(1.0, device=self._torch_dev), self.potential + delta * 0.4
        )
        fired = self.potential >= FIRE_THRESHOLD
        self.activation = torch.where(
            fired,
            torch.minimum(
                torch.tensor(1.0, device=self._torch_dev),
                self.activation + self.potential * 0.35,
            ),
            self.activation * (1.0 - decay),
        )
        self.potential = torch.where(
            fired, self.potential * 0.3, self.potential * (1.0 - decay)
        )

    def _propagate_numpy(self, decay: float) -> None:
        exc = _convolve2d_numpy(self.activation, self._kernel_exc)
        inh = _convolve2d_numpy(self.activation, self._kernel_inh)
        delta = exc - inh * 0.45
        self.potential = np.minimum(1.0, self.potential + delta * 0.4)
        fired = self.potential >= FIRE_THRESHOLD
        self.activation = np.where(
            fired,
            np.minimum(1.0, self.activation + self.potential * 0.35),
            self.activation * (1.0 - decay),
        )
        self.potential = np.where(fired, self.potential * 0.3, self.potential * (1.0 - decay))

    def _propagate_python(self, decay: float) -> None:
        h, w = self.height, self.width
        delta = [[0.0] * w for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                exc = sum(
                    self.activation[y + ky - 1][x + kx - 1] * self._kernel_exc[ky][kx]
                    for ky in range(3)
                    for kx in range(3)
                )
                inh = sum(
                    self.activation[y + ky - 1][x + kx - 1] * self._kernel_inh[ky][kx]
                    for ky in range(3)
                    for kx in range(3)
                )
                delta[y][x] = exc - inh * 0.45
        for y in range(h):
            for x in range(w):
                self.potential[y][x] = min(1.0, self.potential[y][x] + delta[y][x] * 0.4)
                if self.potential[y][x] >= FIRE_THRESHOLD:
                    self.activation[y][x] = min(
                        1.0, self.activation[y][x] + self.potential[y][x] * 0.35
                    )
                    self.potential[y][x] *= 0.3
                else:
                    self.activation[y][x] *= 1.0 - decay
                    self.potential[y][x] *= 1.0 - decay

    def _apply_long_range(self, syn: LongRangeSynapse) -> None:
        if not (0 <= syn.sx < self.width and 0 <= syn.sy < self.height):
            return
        if not (0 <= syn.dx < self.width and 0 <= syn.dy < self.height):
            return
        src = self._get_act(syn.sy, syn.sx)
        if src < FIRE_THRESHOLD:
            return
        self._set_pot(syn.dy, syn.dx, min(1.0, self._get_pot(syn.dy, syn.dx) + src * syn.weight))

    def _get_act(self, y: int, x: int) -> float:
        return scalar(self.activation[y, x] if not isinstance(self.activation[0], list) else self.activation[y][x])

    def _get_pot(self, y: int, x: int) -> float:
        return scalar(self.potential[y, x] if not isinstance(self.potential[0], list) else self.potential[y][x])

    def _set_pot(self, y: int, x: int, v: float) -> None:
        if isinstance(self.potential[0], list):
            self.potential[y][x] = v
        else:
            self.potential[y, x] = v

    def salience_map(self) -> Any:
        if self._is_torch():
            gy, gx = torch.gradient(self.activation)
            grad = torch.sqrt(gx * gx + gy * gy)
            return self.activation * (0.6 + 0.4 * grad)
        if HAS_NUMPY and np is not None and not isinstance(self.activation, list):
            gy, gx = np.gradient(self.activation.astype(np.float32))
            grad = np.sqrt(gx * gx + gy * gy)
            return self.activation * (0.6 + 0.4 * grad)
        h, w = self.height, self.width
        out = [[0.0] * w for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                gx = self.activation[y][x + 1] - self.activation[y][x - 1]
                gy = self.activation[y + 1][x] - self.activation[y - 1][x]
                grad = (gx * gx + gy * gy) ** 0.5
                out[y][x] = self.activation[y][x] * (0.6 + 0.4 * grad)
        return out

    def hotspots(self, k: int = 8, *, min_distance: int = 6) -> list[tuple[int, int, float]]:
        sal = self.salience_map()
        candidates: list[tuple[int, int, float]] = []

        if self._is_torch():
            flat = sal.reshape(-1)
            vals, idxs = torch.topk(flat, min(k * 8, flat.numel()))
            taken = torch.zeros(sal.shape, dtype=torch.bool, device=self._torch_dev)
            for idx_t, val_t in zip(idxs.tolist(), vals.tolist()):
                if len(candidates) >= k or val_t < 0.05:
                    break
                y, x = divmod(int(idx_t), self.width)
                if taken[y, x]:
                    continue
                candidates.append((x, y, float(val_t)))
                y0 = max(0, y - min_distance)
                y1 = min(self.height, y + min_distance + 1)
                x0 = max(0, x - min_distance)
                x1 = min(self.width, x + min_distance + 1)
                taken[y0:y1, x0:x1] = True
            return candidates

        if HAS_NUMPY and np is not None and not isinstance(sal, list):
            flat = sal.ravel()
            order = np.argsort(flat)[::-1]
            taken = np.zeros(sal.shape, dtype=bool)
            for idx in order:
                if len(candidates) >= k:
                    break
                y, x = divmod(int(idx), self.width)
                if taken[y, x]:
                    continue
                v = float(sal[y, x])
                if v < 0.05:
                    break
                candidates.append((x, y, v))
                y0 = max(0, y - min_distance)
                y1 = min(self.height, y + min_distance + 1)
                x0 = max(0, x - min_distance)
                x1 = min(self.width, x + min_distance + 1)
                taken[y0:y1, x0:x1] = True
            return candidates

        for y in range(self.height):
            for x in range(self.width):
                candidates.append((x, y, float(sal[y][x])))
        candidates.sort(key=lambda t: t[2], reverse=True)
        picked: list[tuple[int, int, float]] = []
        for x, y, v in candidates:
            if v < 0.05:
                break
            if any(abs(x - px) < min_distance and abs(y - py) < min_distance for px, py, _ in picked):
                continue
            picked.append((x, y, v))
            if len(picked) >= k:
                break
        return picked

    def active_ratio(self) -> float:
        if self._is_torch():
            return float((self.activation >= FIRE_THRESHOLD).float().mean().item())
        if HAS_NUMPY and np is not None and not isinstance(self.activation, list):
            return float((self.activation >= FIRE_THRESHOLD).mean())
        total = self.neuron_count
        active = sum(
            1
            for y in range(self.height)
            for x in range(self.width)
            if self.activation[y][x] >= FIRE_THRESHOLD
        )
        return active / max(1, total)

    def mean_activation(self) -> float:
        if self._is_torch():
            return float(self.activation.mean().item())
        if HAS_NUMPY and np is not None and not isinstance(self.activation, list):
            return float(self.activation.mean())
        s = sum(self.activation[y][x] for y in range(self.height) for x in range(self.width))
        return s / max(1, self.neuron_count)

    def to_activation_grid(self) -> list[list[float]]:
        if self._is_torch():
            return self.activation.detach().cpu().tolist()
        if HAS_NUMPY and np is not None and not isinstance(self.activation, list):
            return self.activation.tolist()
        return [[float(self.activation[y][x]) for x in range(self.width)] for y in range(self.height)]

    def to_activation_bytes(self) -> bytes:
        """Griglia 0-255 per UI canvas — utile per preview locale."""
        grid = self.to_activation_grid()
        out = bytearray(self.width * self.height)
        for y, row in enumerate(grid):
            for x, v in enumerate(row):
                out[y * self.width + x] = max(0, min(255, int(v * 255)))
        return bytes(out)

    def stats(self) -> dict[str, Any]:
        st = {
            "width": self.width,
            "height": self.height,
            "neurons": self.neuron_count,
            "tick": self.tick,
            "active_ratio": round(self.active_ratio(), 5),
            "mean_activation": round(self.mean_activation(), 5),
            "long_range_synapses": len(self.long_range),
            "backend": self._backend,
            "uses_gpu": self.uses_gpu,
        }
        if self.uses_gpu and HAS_TORCH:
            st["gpu_name"] = torch.cuda.get_device_name(self._torch_dev)
        return st

    def _is_torch(self) -> bool:
        return self._backend != "numpy" and HAS_TORCH and self._torch_dev is not None


def create_retina_cortex(
    width: int,
    height: int,
    *,
    device: str = "auto",
) -> RetinaCortex:
    """Factory — ``device=auto`` sceglie CUDA su Windows se disponibile."""
    return RetinaCortex(width=width, height=height, device=device)


def _convolve2d_numpy(field: Any, kernel: Any) -> Any:
    if not HAS_NUMPY or np is None:
        raise RuntimeError("numpy required")
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(field, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    out = np.zeros_like(field, dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            out += kernel[ky, kx] * padded[ky : ky + field.shape[0], kx : kx + field.shape[1]]
    return out


__all__ = [
    "RetinaCortex",
    "LongRangeSynapse",
    "create_retina_cortex",
    "FIRE_THRESHOLD",
    "cuda_available",
    "gpu_info",
]
