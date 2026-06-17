"""Retina cortex — ogni pixel è un neurone, sinapsi virtuali via kernel locale.

Non memorizziamo milioni × milioni di sinapsi esplicite: ogni neurone (x, y)
è collegato ai vicini tramite un kernel convolutivo (campo recettivo locale).
Le connessioni a lungo raggio sono sparse e opzionali.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False

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
    activation: Any = field(init=False, repr=False)
    potential: Any = field(init=False, repr=False)
    tick: float = 0.0
    long_range: list[LongRangeSynapse] = field(default_factory=list)
    _kernel_exc: Any = field(init=False, repr=False)
    _kernel_inh: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width < 4 or self.height < 4:
            raise ValueError("retina troppo piccola (min 4×4)")
        if HAS_NUMPY and np is not None:
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
        fired = 0
        if HAS_NUMPY and np is not None:
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
            fired = int(mask.sum())
        else:
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
        """Stimolo puntuale — un singolo neurone-pixel."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        if HAS_NUMPY and np is not None:
            self.potential[y, x] = min(1.0, float(self.potential[y, x]) + intensity)
            if self.potential[y, x] >= FIRE_THRESHOLD:
                self.activation[y, x] = min(1.0, float(self.activation[y, x]) + intensity * 0.6)
        else:
            self.potential[y][x] = min(1.0, self.potential[y][x] + intensity)
            if self.potential[y][x] >= FIRE_THRESHOLD:
                self.activation[y][x] = min(1.0, self.activation[y][x] + intensity * 0.6)

    def propagate(self, steps: int = 1, *, decay: float = LEAK_RATE) -> None:
        """Diffusione sinaptica locale + sinapsi sparse a lungo raggio."""
        for _ in range(steps):
            self.tick += 1.0
            if HAS_NUMPY and np is not None:
                exc = _convolve2d(self.activation, self._kernel_exc)
                inh = _convolve2d(self.activation, self._kernel_inh)
                delta = exc - inh * 0.45
                self.potential = np.minimum(1.0, self.potential + delta * 0.4)
                fired = self.potential >= FIRE_THRESHOLD
                self.activation = np.where(
                    fired,
                    np.minimum(1.0, self.activation + self.potential * 0.35),
                    self.activation * (1.0 - decay),
                )
                self.potential = np.where(fired, self.potential * 0.3, self.potential * (1.0 - decay))
            else:
                self._propagate_python(decay)

            for syn in self.long_range:
                self._apply_long_range(syn)

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
        dst = self._get_pot(syn.dy, syn.dx) + src * syn.weight
        self._set_pot(syn.dy, syn.dx, min(1.0, dst))

    def _get_act(self, y: int, x: int) -> float:
        if HAS_NUMPY and np is not None:
            return float(self.activation[y, x])
        return float(self.activation[y][x])

    def _get_pot(self, y: int, x: int) -> float:
        if HAS_NUMPY and np is not None:
            return float(self.potential[y, x])
        return float(self.potential[y][x])

    def _set_pot(self, y: int, x: int, v: float) -> None:
        if HAS_NUMPY and np is not None:
            self.potential[y, x] = v
        else:
            self.potential[y][x] = v

    def salience_map(self) -> Any:
        """Dove il cervello 'brilla' — attivazione × gradiente locale."""
        if HAS_NUMPY and np is not None:
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
        """Punti precisi di massima attività — coordinate che la coscienza può leggere."""
        sal = self.salience_map()
        candidates: list[tuple[int, int, float]] = []
        if HAS_NUMPY and np is not None:
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
        else:
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
            candidates = picked
        return candidates

    def active_ratio(self) -> float:
        if HAS_NUMPY and np is not None:
            return float((self.activation >= FIRE_THRESHOLD).mean())
        total = self.neuron_count
        active = sum(
            1 for y in range(self.height) for x in range(self.width) if self.activation[y][x] >= FIRE_THRESHOLD
        )
        return active / max(1, total)

    def mean_activation(self) -> float:
        if HAS_NUMPY and np is not None:
            return float(self.activation.mean())
        s = sum(self.activation[y][x] for y in range(self.height) for x in range(self.width))
        return s / max(1, self.neuron_count)

    def to_activation_grid(self) -> list[list[float]]:
        """Esporta attivazioni come griglia leggibile (per UI / debug)."""
        if HAS_NUMPY and np is not None:
            return self.activation.tolist()
        return [[float(self.activation[y][x]) for x in range(self.width)] for y in range(self.height)]

    def stats(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "neurons": self.neuron_count,
            "tick": self.tick,
            "active_ratio": round(self.active_ratio(), 5),
            "mean_activation": round(self.mean_activation(), 5),
            "long_range_synapses": len(self.long_range),
        }


def _convolve2d(field: Any, kernel: Any) -> Any:
    """Convoluzione 2D con padding zero — sinapsi virtuali locali."""
    if not HAS_NUMPY or np is None:
        raise RuntimeError("numpy required for fast convolution")
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(field, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    out = np.zeros_like(field, dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            out += kernel[ky, kx] * padded[ky : ky + field.shape[0], kx : kx + field.shape[1]]
    return out
