"""Lightweight array helpers — numpy if present, else pure Python."""

from __future__ import annotations

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False


def grayscale_grid(data: list[list[int]] | bytes, width: int, height: int) -> list[list[float]]:
    if isinstance(data, bytes):
        pixels = list(data[: width * height])
    else:
        pixels = [p for row in data for p in row]
    grid: list[list[float]] = []
    for y in range(height):
        row = []
        for x in range(width):
            v = pixels[y * width + x] if y * width + x < len(pixels) else 0
            row.append(float(v) / 255.0)
        grid.append(row)
    return grid


def sobel_edges(grid: list[list[float]]) -> list[list[float]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    out = [[0.0] * w for _ in range(h)]
    gx_k = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    gy_k = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = gy = 0.0
            for ky in range(3):
                for kx in range(3):
                    v = grid[y + ky - 1][x + kx - 1]
                    gx += gx_k[ky][kx] * v
                    gy += gy_k[ky][kx] * v
            out[y][x] = min(1.0, (gx * gx + gy * gy) ** 0.5)
    return out


def fft_band_energy(samples: list[float], sample_rate: int, low: float, high: float) -> float:
    if HAS_NUMPY and np is not None:
        arr = np.array(samples, dtype=float)
        if len(arr) == 0:
            return 0.0
        fft = np.fft.rfft(arr)
        freqs = np.fft.rfftfreq(len(arr), d=1.0 / sample_rate)
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(np.abs(fft[mask])))
    # naive DFT fallback for tiny buffers in tests
    n = len(samples)
    if n == 0:
        return 0.0
    energy = 0.0
    for k in range(n // 2):
        f = k * sample_rate / n
        if low <= f < high:
            re = sum(samples[t] * __import__("math").cos(2 * 3.14159 * f * t / sample_rate) for t in range(n))
            im = sum(samples[t] * __import__("math").sin(2 * 3.14159 * f * t / sample_rate) for t in range(n))
            energy += (re * re + im * im) ** 0.5
    return energy / max(1, n)
