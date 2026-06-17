"""GPUBrainEngine — pipeline triple-buffer locale (no server)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mindruntime import cuda_util
from mindruntime.gpu_core import (
    initialize_neurons,
    match_resonators,
    propagate_wavefront,
    update_weights_hebbian,
)
from mindruntime.resonators import TEMPLATE_NAMES, build_resonator_bank


@dataclass
class EngineStats:
    tick: int = 0
    width: int = 0
    height: int = 0
    backend: str = "cpu"
    fps: float = 0.0
    last_recognition: list[tuple[str, float]] = field(default_factory=list)


class GPUBrainEngine:
    """Simulatore cerebrale GPU — webcam/immagine → onde → simboli."""

    def __init__(
        self,
        *,
        width: int = 256,
        height: int = 256,
        hebb_every: int = 8,
        match_every: int = 6,
        seed: int = 42,
    ) -> None:
        if width < 32 or height < 32:
            raise ValueError("risoluzione minima 32×32")
        self.width = width
        self.height = height
        self.hebb_every = hebb_every
        self.match_every = match_every
        self.seed = seed
        self._initialized = False
        self._stats = EngineStats(width=width, height=height)
        self._bank = build_resonator_bank(TEMPLATE_NAMES)
        self._tpl_stack = self._bank["stack"]
        self._tpl_names: list[str] = self._bank["names"]
        self._buffers = self._alloc_buffers()
        self._last_frame_time = time.perf_counter()
        info = cuda_util.cuda_info()
        self._stats.backend = "cuda" if info.get("cuda") else "cpu"

    @property
    def stats(self) -> EngineStats:
        return self._stats

    @property
    def uses_cuda(self) -> bool:
        return self._stats.backend == "cuda"

    def _alloc_buffers(self) -> list[np.ndarray]:
        shape = (self.height, self.width, 4)
        return [np.zeros(shape, dtype=np.float32) for _ in range(3)]

    @property
    def current(self) -> np.ndarray:
        return self._buffers[0]

    def step(self, input_frame: np.ndarray | None = None) -> dict[str, Any]:
        """Un tick fisico. Primo frame o nuovo input → reinizializza impulsi."""
        t0 = time.perf_counter()
        if input_frame is not None:
            frame = self._resize_rgb(input_frame)
            if not self._initialized:
                initialize_neurons(frame, self._buffers[0], seed=self.seed)
                self._buffers[1][:] = self._buffers[0]
                self._buffers[2][:] = self._buffers[0]
                self._initialized = True
            else:
                # reiniezione debole da nuovo frame (retina in movimento)
                self._inject_frame(frame, gain=0.35)

        if not self._initialized:
            raise RuntimeError("chiama step() con un frame prima del loop")

        # buf[0]=t, buf[1]=t-1, buf[2]=t-2 → nuovo t in scratch, poi shift
        new_state = np.zeros_like(self._buffers[0])
        propagate_wavefront(self._buffers[1], self._buffers[2], new_state)
        self._buffers[2] = self._buffers[1]
        self._buffers[1] = self._buffers[0]
        self._buffers[0] = new_state

        self._stats.tick += 1
        if self._stats.tick % self.hebb_every == 0:
            update_weights_hebbian(self._buffers[0])

        if self._stats.tick % self.match_every == 0:
            self._stats.last_recognition = self._match_symbols()

        dt = time.perf_counter() - t0
        if dt > 0:
            self._stats.fps = 0.9 * self._stats.fps + 0.1 * (1.0 / dt)

        return {
            "tick": self._stats.tick,
            "recognition": list(self._stats.last_recognition),
            "backend": self._stats.backend,
            "fps": round(self._stats.fps, 1),
        }

    def render(self) -> np.ndarray:
        """Mappa fase → RGB uint8 per visualizzazione."""
        phase = self._buffers[0][:, :, 1]
        impulse = self._buffers[0][:, :, 0]
        weight = self._buffers[0][:, :, 2]
        h = (phase / (2 * np.pi)) % 1.0
        s = np.clip(impulse * 0.85 + 0.15, 0, 1)
        v = np.clip(0.35 + impulse * 0.45 + weight * 0.12, 0, 1)
        rgb = _hsv_to_rgb(h, s, v)
        return (rgb * 255).astype(np.uint8)

    def render_overlay(self) -> np.ndarray:
        """Immagine con barra testo riconoscimento (per OpenCV)."""
        img = self.render().copy()
        y = 18
        for sym, score in self._stats.last_recognition[:5]:
            _draw_label(img, f"{sym}:{score:.2f}", y)
            y += 16
        _draw_label(img, f"{'CUDA' if self.uses_cuda else 'CPU'} tick={self._stats.tick} fps={self._stats.fps:.0f}", y)
        return img

    def export_state_for_training(self) -> dict[str, np.ndarray]:
        """Serializza stato per PyTorch/JAX (canale AI training)."""
        cur = self._buffers[0]
        return {
            "impulse": cur[:, :, 0].copy(),
            "phase": cur[:, :, 1].copy(),
            "weight": cur[:, :, 2].copy(),
            "velocity_cache": cur[:, :, 3].copy(),
            "history_t1": self._buffers[1][:, :, 0].copy(),
            "history_t2": self._buffers[2][:, :, 0].copy(),
            "tick": np.array([self._stats.tick], dtype=np.int32),
        }

    def apply_weight_delta(self, delta: np.ndarray) -> None:
        """Applica gradiente esterno sui pesi (da AITrainer)."""
        d = np.asarray(delta, dtype=np.float32)
        if d.shape != (self.height, self.width):
            raise ValueError("delta pesi deve essere H×W")
        self._buffers[0][:, :, 2] = np.clip(self._buffers[0][:, :, 2] + d, 0.02, 3.0)
        self._buffers[1][:, :, 2] = self._buffers[0][:, :, 2]
        self._buffers[2][:, :, 2] = self._buffers[0][:, :, 2]

    def update_resonators(self, new_stack: np.ndarray, names: list[str] | None = None) -> None:
        if new_stack.ndim != 3:
            raise ValueError("new_stack deve essere N×H×W")
        self._tpl_stack = np.ascontiguousarray(new_stack.astype(np.float32))
        if names:
            self._tpl_names = list(names)

    def _match_symbols(self, top_k: int = 5) -> list[tuple[str, float]]:
        imp = self._buffers[0][:, :, 0]
        scores = match_resonators(imp, self._tpl_stack)
        order = np.argsort(scores)[::-1]
        out: list[tuple[str, float]] = []
        for idx in order[:top_k]:
            sc = float(scores[idx])
            if sc > 0.08:
                out.append((self._tpl_names[int(idx)], sc))
        return out

    def _resize_rgb(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = _resize_bilinear(frame, self.height, self.width)
        if frame.dtype == np.uint8:
            return frame.astype(np.float32) / 255.0
        return frame.astype(np.float32)

    def _inject_frame(self, rgb: np.ndarray, *, gain: float) -> None:
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        mask = lum > 0.18
        self._buffers[0][mask, 0] = np.minimum(
            1.0, self._buffers[0][mask, 0] + lum[mask] * gain
        )
        self._buffers[0][mask, 1] = (self._buffers[0][mask, 1] + lum[mask] * np.pi * 0.25) % (
            2 * np.pi
        )


def _resize_bilinear(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = img.shape[:2]
    ys = (np.linspace(0, h - 1, th)).astype(np.float32)
    xs = (np.linspace(0, w - 1, tw)).astype(np.float32)
    out = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
    for yi, yf in enumerate(ys):
        y0, y1 = int(yf), min(int(yf) + 1, h - 1)
        wy = yf - int(yf)
        for xi, xf in enumerate(xs):
            x0, x1 = int(xf), min(int(xf) + 1, w - 1)
            wx = xf - int(xf)
            for c in range(img.shape[2]):
                v = (
                    img[y0, x0, c] * (1 - wx) * (1 - wy)
                    + img[y0, x1, c] * wx * (1 - wy)
                    + img[y1, x0, c] * (1 - wx) * wy
                    + img[y1, x1, c] * wx * wy
                )
                out[yi, xi, c] = v
    return out


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _draw_label(img: np.ndarray, text: str, y: int) -> None:
    """Testo minimale senza dipendere da cv2.putText."""
    x0 = 6
    for i, ch in enumerate(text[:28]):
        _stamp_char(img, ch, x0 + i * 7, y)


def _stamp_char(img: np.ndarray, ch: str, x: int, y: int) -> None:
    if y < 0 or y + 8 >= img.shape[0] or x < 0 or x + 6 >= img.shape[1]:
        return
    val = min(255, ord(ch))
    img[y : y + 8, x : x + 6, 1] = np.clip(img[y : y + 8, x : x + 6, 1] + val % 40, 0, 255)
