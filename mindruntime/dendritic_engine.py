"""DendriticBrainEngine — compartimenti ionici + loop backward, solo locale."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mindruntime import cuda_util
from mindruntime.dendritic_core import (
    CH_BW,
    CH_CA,
    CH_IMP,
    CH_K,
    CH_NA,
    CH_PH,
    CH_W,
    N_CHANNELS,
    backward_dendrite,
    coherence_map,
    forward_dendrite,
    initialize_dendrites,
    match_resonators,
)
from mindruntime.gpu_engine import _hsv_to_rgb, _resize_bilinear
from mindruntime.resonators import TEMPLATE_NAMES, build_resonator_bank


@dataclass
class DendriticStats:
    tick: int = 0
    width: int = 0
    height: int = 0
    backend: str = "cpu"
    fps: float = 0.0
    mean_coherence: float = 0.0
    last_recognition: list[tuple[str, float]] = field(default_factory=list)


class DendriticBrainEngine:
    """Cervello dendritico emergente — Na/K/Ca + backward + risonatori."""

    def __init__(
        self,
        *,
        width: int = 256,
        height: int = 256,
        backward_every: int = 4,
        match_every: int = 6,
        seed: int = 42,
    ) -> None:
        if width < 32 or height < 32:
            raise ValueError("risoluzione minima 32×32")
        self.width = width
        self.height = height
        self.backward_every = backward_every
        self.match_every = match_every
        self.seed = seed
        self._initialized = False
        self._stats = DendriticStats(width=width, height=height)
        self._bank = build_resonator_bank(TEMPLATE_NAMES)
        self._tpl_stack = self._bank["stack"]
        self._tpl_names: list[str] = self._bank["names"]
        self._buffers = [np.zeros((height, width, N_CHANNELS), dtype=np.float32) for _ in range(3)]
        info = cuda_util.cuda_info()
        self._stats.backend = "cuda" if info.get("cuda") else "cpu"

    @property
    def stats(self) -> DendriticStats:
        return self._stats

    @property
    def uses_cuda(self) -> bool:
        return self._stats.backend == "cuda"

    def step(self, input_frame: np.ndarray | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        if input_frame is not None:
            frame = self._resize_rgb(input_frame)
            if not self._initialized:
                initialize_dendrites(frame, self._buffers[0], seed=self.seed)
                self._buffers[1][:] = self._buffers[0]
                self._buffers[2][:] = self._buffers[0]
                self._initialized = True
            else:
                self._inject_frame(frame, gain=0.32)

        if not self._initialized:
            raise RuntimeError("chiama step() con un frame prima del loop")

        new_state = np.zeros_like(self._buffers[0])
        forward_dendrite(self._buffers[1], self._buffers[2], new_state)
        self._buffers[2] = self._buffers[1]
        self._buffers[1] = self._buffers[0]
        self._buffers[0] = new_state

        self._stats.tick += 1
        if self._stats.tick % self.backward_every == 0:
            backward_dendrite(self._buffers[0])
        if self._stats.tick % self.match_every == 0:
            self._stats.last_recognition = self._match_symbols()

        coh = coherence_map(self._buffers[0])
        self._stats.mean_coherence = float(coh.mean())

        dt = time.perf_counter() - t0
        if dt > 0:
            self._stats.fps = 0.9 * self._stats.fps + 0.1 * (1.0 / dt)

        return {
            "tick": self._stats.tick,
            "coherence": round(self._stats.mean_coherence, 4),
            "recognition": list(self._stats.last_recognition),
            "backend": self._stats.backend,
            "fps": round(self._stats.fps, 1),
        }

    def render(self) -> np.ndarray:
        """Ca²⁺ + coerenza → colore; peso → luminosità."""
        cur = self._buffers[0]
        phase = cur[:, :, CH_PH]
        ca = cur[:, :, CH_CA]
        coh = coherence_map(cur)
        weight = cur[:, :, CH_W]
        h = (phase / (2 * np.pi) + ca * 0.2) % 1.0
        s = np.clip(0.2 + coh * 0.9, 0, 1)
        v = np.clip(0.3 + coh * 0.5 + weight * 0.15, 0, 1)
        rgb = _hsv_to_rgb(h, s, v)
        return (rgb * 255).astype(np.uint8)

    def render_composite(self, camera_bgr: np.ndarray | None = None) -> np.ndarray:
        """Vista principale cervello + inset webcam (no browser)."""
        brain = self.render()
        if camera_bgr is not None:
            import cv2

            th, tw = max(64, self.height // 5), max(80, self.width // 5)
            inset = cv2.resize(camera_bgr, (tw, th))
            brain_bgr = cv2.cvtColor(brain, cv2.COLOR_RGB2BGR)
            y0, x0 = 8, brain_bgr.shape[1] - tw - 8
            brain_bgr[y0 : y0 + th, x0 : x0 + tw] = inset
            cv2.rectangle(brain_bgr, (x0 - 1, y0 - 1), (x0 + tw, y0 + th), (40, 220, 180), 1)
            return brain_bgr
        return brain[:, :, ::-1]  # RGB→BGR

    def overlay_lines(self) -> list[str]:
        lines = [
            f"{'CUDA' if self.uses_cuda else 'CPU'} dendrite tick={self._stats.tick} fps={self._stats.fps:.0f}",
            f"coerenza={self._stats.mean_coherence:.3f}",
        ]
        for sym, sc in self._stats.last_recognition[:4]:
            lines.append(f"{sym}:{sc:.2f}")
        return lines

    def export_state_for_training(self) -> dict[str, np.ndarray]:
        cur = self._buffers[0]
        return {
            "impulse": cur[:, :, CH_IMP].copy(),
            "phase": cur[:, :, CH_PH].copy(),
            "weight": cur[:, :, CH_W].copy(),
            "na": cur[:, :, CH_NA].copy(),
            "k": cur[:, :, CH_K].copy(),
            "ca": cur[:, :, CH_CA].copy(),
            "backward": cur[:, :, CH_BW].copy(),
            "coherence": coherence_map(cur),
            "tick": np.array([self._stats.tick], dtype=np.int32),
        }

    def _match_symbols(self, top_k: int = 5) -> list[tuple[str, float]]:
        # matching su pattern di coerenza, non solo impulso grezzo
        field = coherence_map(self._buffers[0])
        scores = match_resonators(field, self._tpl_stack)
        order = np.argsort(scores)[::-1]
        out: list[tuple[str, float]] = []
        for idx in order[:top_k]:
            sc = float(scores[idx])
            if sc > 0.07:
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
        mask = lum > 0.16
        self._buffers[0][mask, CH_IMP] = np.minimum(
            1.0, self._buffers[0][mask, CH_IMP] + lum[mask] * gain
        )
        self._buffers[0][mask, CH_NA] = np.minimum(
            1.0, self._buffers[0][mask, CH_NA] + lum[mask] * gain * 0.4
        )
