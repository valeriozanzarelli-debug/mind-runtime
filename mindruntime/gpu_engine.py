"""GPUBrainEngine — pipeline fisica emergente (Hodgkin-Huxley + Turing + SOC)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mindruntime import cuda_util
from mindruntime.gpu_physics import (
    CH_CA,
    CH_COH,
    CH_EN,
    CH_IMP,
    CH_K,
    CH_NA,
    CH_PH,
    CH_W,
    N_CHANNELS,
    bifurcation_phase_lock,
    hodgkin_huxley_step,
    initialize_field,
    initialize_hh_state,
    phase_gradient_memory,
    soc_criticality_tunning,
    turing_reaction_diffusion,
)
from mindruntime.resonators import correlate_resonators, load_resonators_from_disk


@dataclass
class EngineStats:
    tick: int = 0
    width: int = 0
    height: int = 0
    backend: str = "cpu"
    fps: float = 0.0
    mean_coherence: float = 0.0
    last_recognition: list[tuple[str, float]] = field(default_factory=list)


class GPUBrainEngine:
    """Simulatore cerebrale GPU — webcam → fisica emergente → simboli."""

    def __init__(
        self,
        *,
        width: int = 256,
        height: int = 256,
        device: int = 0,
        soc_every: int = 10,
        memory_every: int = 20,
        hh_substeps: int = 3,
        seed: int = 42,
    ) -> None:
        if width < 32 or height < 32:
            raise ValueError("risoluzione minima 32×32")
        self.width = width
        self.height = height
        self.device = device
        self.soc_every = soc_every
        self.memory_every = memory_every
        self.hh_substeps = hh_substeps
        self.seed = seed
        self.step_count = 0
        self.is_first_frame = True
        self._stats = EngineStats(width=width, height=height)
        self._resonators = load_resonators_from_disk()
        self._buffers = self._alloc_buffers()
        info = cuda_util.cuda_info()
        self._stats.backend = "cuda" if info.get("cuda") else "cpu"

    @property
    def stats(self) -> EngineStats:
        return self._stats

    @property
    def uses_cuda(self) -> bool:
        return self._stats.backend == "cuda"

    @property
    def field_t(self) -> np.ndarray:
        return self._buffers[0]

    def _alloc_buffers(self) -> list[np.ndarray]:
        shape = (self.height, self.width, N_CHANNELS)
        return [np.zeros(shape, dtype=np.float32) for _ in range(3)]

    def step(self, input_frame: np.ndarray | None = None) -> dict[str, Any]:
        """Un passo di simulazione cervello."""
        t0 = time.perf_counter()
        if input_frame is not None:
            frame = self._resize_rgb(input_frame)
            if self.is_first_frame:
                initialize_field(frame, self._buffers[0], threshold=0.3, seed=self.seed)
                initialize_hh_state(self._hh_state)
                self._buffers[1][:] = self._buffers[0]
                self._buffers[2][:] = self._buffers[0]
                self.is_first_frame = False
            else:
                self._inject_frame(frame, gain=0.3)

        if self.is_first_frame:
            raise RuntimeError("chiama step() con un frame prima del loop")

        field = self._buffers[0]
        for _ in range(self.hh_substeps):
            hodgkin_huxley_step(field, self._hh_state, dt=0.001)

        new_state = np.zeros_like(field)
        turing_reaction_diffusion(
            field, self._buffers[1], self._buffers[2], self._hh_state, new_state, decay=0.97
        )
        self._buffers[2] = self._buffers[1]
        self._buffers[1] = self._buffers[0]
        self._buffers[0] = new_state
        field = self._buffers[0]

        if self.step_count % self.soc_every == 0:
            soc_criticality_tunning(field, self._avalanche_map)

        bifurcation_phase_lock(field)

        if self.step_count % self.memory_every == 0:
            phase_gradient_memory(field, self._persistence)

        self.step_count += 1
        self._stats.tick = self.step_count
        self._stats.mean_coherence = float(field[:, :, CH_COH].mean())
        self._stats.last_recognition = self.get_recognition(confidence_thresh=0.07)

        dt = time.perf_counter() - t0
        if dt > 0:
            self._stats.fps = 0.9 * self._stats.fps + 0.1 * (1.0 / dt)

        return {
            "tick": self.step_count,
            "coherence": round(self._stats.mean_coherence, 4),
            "recognition": list(self._stats.last_recognition),
            "backend": self._stats.backend,
            "fps": round(self._stats.fps, 1),
        }

    def render(self) -> np.ndarray:
        """Mappa fase → hue, impulse → saturation, coherence → value."""
        field = self._buffers[0]
        phase = field[:, :, CH_PH]
        impulse = field[:, :, CH_IMP]
        coherence = field[:, :, CH_COH]
        h = (phase / TWO_PI) % 1.0
        s = np.clip(impulse, 0, 1)
        v = np.clip(coherence * 0.85 + 0.15, 0, 1)
        rgb = _hsv_to_rgb(h, s, v)
        return (rgb * 255).astype(np.uint8)

    def render_overlay(self) -> np.ndarray:
        """Immagine con etichette riconoscimento."""
        img = self.render().copy()
        y = 18
        for sym, score in self._stats.last_recognition[:5]:
            _draw_label(img, f"{sym}:{score:.2f}", y)
            y += 16
        _draw_label(
            img,
            f"{'CUDA' if self.uses_cuda else 'CPU'} tick={self.step_count} fps={self._stats.fps:.0f}",
            y,
        )
        return img

    def get_recognition(self, *, confidence_thresh: float = 0.5) -> list[tuple[str, float]]:
        """Estrai pattern riconosciuti via correlazione FFT risonatori."""
        coherence_map = self._buffers[0][:, :, CH_COH].astype(np.float32)
        recognitions = correlate_resonators(coherence_map, self._resonators)
        return [
            (sym, conf) for sym, conf in recognitions if conf > confidence_thresh
        ][:5]

    def export_state_for_server(self) -> dict[str, Any]:
        """Serializza stato GPU per salvataggio remoto."""
        return {
            "field": self._buffers[0].tobytes(),
            "hh_state": self._hh_state.tobytes(),
            "shape": (self.height, self.width, N_CHANNELS),
            "step_count": self.step_count,
            "timestamp": time.time(),
        }

    def import_state_from_server(self, state_dict: dict[str, Any]) -> None:
        """Ripristina stato da server."""
        shape = tuple(state_dict["shape"])
        self._buffers[0] = np.frombuffer(state_dict["field"], dtype=np.float32).reshape(shape).copy()
        if "hh_state" in state_dict:
            self._hh_state = np.frombuffer(state_dict["hh_state"], dtype=np.float32).reshape(
                self.height, self.width, 4
            ).copy()
        self.step_count = int(state_dict.get("step_count", 0))
        self.is_first_frame = False

    def export_state_for_training(self) -> dict[str, np.ndarray]:
        """Compatibilità AITrainer — mappa canali fisici."""
        cur = self._buffers[0]
        return {
            "impulse": cur[:, :, CH_IMP].copy(),
            "phase": cur[:, :, CH_PH].copy(),
            "weight": cur[:, :, CH_W].copy(),
            "velocity_cache": cur[:, :, CH_EN].copy(),
            "calcium": cur[:, :, CH_CA].copy(),
            "sodium": cur[:, :, CH_NA].copy(),
            "potassium": cur[:, :, CH_K].copy(),
            "coherence": cur[:, :, CH_COH].copy(),
            "history_t1": self._buffers[1][:, :, CH_IMP].copy(),
            "history_t2": self._buffers[2][:, :, CH_IMP].copy(),
            "tick": np.array([self.step_count], dtype=np.int32),
        }

    def apply_weight_delta(self, delta: np.ndarray) -> None:
        """Applica gradiente esterno sui pesi."""
        d = np.asarray(delta, dtype=np.float32)
        if d.shape != (self.height, self.width):
            raise ValueError("delta pesi deve essere H×W")
        self._buffers[0][:, :, CH_W] = np.clip(self._buffers[0][:, :, CH_W] + d, 0.02, 3.0)

    def update_resonators(self, resonators: list[dict[str, Any]]) -> None:
        self._resonators = resonators

    def overlay_lines(self) -> list[str]:
        lines = [
            f"{'CUDA' if self.uses_cuda else 'CPU'} physics tick={self.step_count} fps={self._stats.fps:.0f}",
            f"coerenza={self._stats.mean_coherence:.3f}",
        ]
        for sym, sc in self._stats.last_recognition[:4]:
            lines.append(f"{sym}:{sc:.2f}")
        return lines

    def render_composite(self, camera_bgr: np.ndarray | None = None) -> np.ndarray:
        """Vista cervello + inset webcam."""
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
        return brain[:, :, ::-1]

    @property
    def _hh_state(self) -> np.ndarray:
        if not hasattr(self, "_hh_state_buf"):
            self._hh_state_buf = np.zeros((self.height, self.width, 4), dtype=np.float32)
            initialize_hh_state(self._hh_state_buf)
        return self._hh_state_buf

    @_hh_state.setter
    def _hh_state(self, value: np.ndarray) -> None:
        self._hh_state_buf = value

    @property
    def _avalanche_map(self) -> np.ndarray:
        if not hasattr(self, "_avalanche_buf"):
            self._avalanche_buf = np.zeros((self.height, self.width), dtype=np.float32)
        return self._avalanche_buf

    @property
    def _persistence(self) -> np.ndarray:
        if not hasattr(self, "_persistence_buf"):
            self._persistence_buf = np.zeros((self.height, self.width), dtype=np.float32)
        return self._persistence_buf

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
        mask = lum > 0.3
        self._buffers[0][mask, CH_IMP] = np.minimum(
            1.0, self._buffers[0][mask, CH_IMP] + lum[mask] * gain
        )
        self._buffers[0][mask, CH_NA] = np.minimum(
            1.0, self._buffers[0][mask, CH_NA] + lum[mask] * gain * 0.2
        )


TWO_PI = 6.283185307179586


def _resize_bilinear(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = img.shape[:2]
    ys = np.linspace(0, h - 1, th).astype(np.float32)
    xs = np.linspace(0, w - 1, tw).astype(np.float32)
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
    x0 = 6
    for i, ch in enumerate(text[:28]):
        _stamp_char(img, ch, x0 + i * 7, y)


def _stamp_char(img: np.ndarray, ch: str, x: int, y: int) -> None:
    if y < 0 or y + 8 >= img.shape[0] or x < 0 or x + 6 >= img.shape[1]:
        return
    val = min(255, ord(ch))
    img[y : y + 8, x : x + 6, 1] = np.clip(img[y : y + 8, x : x + 6, 1] + val % 40, 0, 255)
