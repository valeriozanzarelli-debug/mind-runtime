"""DendriticBrainEngine — dendriti + fisica emergente (Turing, SOC, coscienza)."""

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
)
from mindruntime.gpu_engine import _hsv_to_rgb, _resize_bilinear
from mindruntime.physics_core import (
    PhysicsState,
    build_phase_templates,
    physics_tick,
)
from mindruntime.resonators import TEMPLATE_NAMES, build_resonator_bank


@dataclass
class DendriticStats:
    tick: int = 0
    width: int = 0
    height: int = 0
    backend: str = "cpu"
    fps: float = 0.0
    mean_coherence: float = 0.0
    order_parameter: float = 0.0
    conscious: bool = False
    phase_transition: str = "subcritical"
    avalanche: float = 0.0
    soc_coupling: float = 0.12
    lock_in: float = 0.0
    locked_symbol: str = ""
    turing_energy: float = 0.0
    last_recognition: list[tuple[str, float]] = field(default_factory=list)


class DendriticBrainEngine:
    """Cervello emergente — Na/K/Ca + Turing + SOC + lock-in di fase."""

    def __init__(
        self,
        *,
        width: int = 256,
        height: int = 256,
        backward_every: int = 4,
        seed: int = 42,
    ) -> None:
        if width < 32 or height < 32:
            raise ValueError("risoluzione minima 32×32")
        self.width = width
        self.height = height
        self.backward_every = backward_every
        self.seed = seed
        self._initialized = False
        self._stats = DendriticStats(width=width, height=height)
        self._bank = build_resonator_bank(TEMPLATE_NAMES)
        self._tpl_names: list[str] = self._bank["names"]
        self._phase_templates = build_phase_templates(self._bank["stack"])
        self._buffers = [np.zeros((height, width, N_CHANNELS), dtype=np.float32) for _ in range(3)]
        self._turing_u = np.ones((height, width), dtype=np.float32) * 0.5
        self._turing_v = np.ones((height, width), dtype=np.float32) * 0.25
        self._physics = PhysicsState()
        self._prev_imp = np.zeros((height, width), dtype=np.float32)
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

        prev_imp = self._buffers[0][:, :, CH_IMP].copy()
        new_state = np.zeros_like(self._buffers[0])
        forward_dendrite(
            self._buffers[1],
            self._buffers[2],
            new_state,
            decay=0.94 - self._physics.soc_coupling * 0.05,
        )
        self._buffers[2] = self._buffers[1]
        self._buffers[1] = self._buffers[0]
        self._buffers[0] = new_state

        self._stats.tick += 1
        if self._stats.tick % self.backward_every == 0:
            backward_dendrite(self._buffers[0])

        phys_out = physics_tick(
            self._buffers[0],
            prev_imp,
            self._turing_u,
            self._turing_v,
            self._physics,
            self._phase_templates,
            self._tpl_names,
        )

        coh = coherence_map(self._buffers[0])
        self._stats.mean_coherence = float(coh.mean())
        self._stats.order_parameter = float(phys_out["order"])
        self._stats.conscious = bool(phys_out["conscious"])
        self._stats.phase_transition = str(phys_out["phase"])
        self._stats.avalanche = float(phys_out["avalanche"])
        self._stats.soc_coupling = float(phys_out["coupling"])
        self._stats.lock_in = float(phys_out["lock_in"])
        self._stats.locked_symbol = str(phys_out["symbol"])
        self._stats.turing_energy = float(phys_out["turing_energy"])
        self._stats.last_recognition = list(phys_out["recognition"])  # type: ignore[arg-type]

        dt = time.perf_counter() - t0
        if dt > 0:
            self._stats.fps = 0.9 * self._stats.fps + 0.1 * (1.0 / dt)

        return {
            "tick": self._stats.tick,
            "coherence": round(self._stats.mean_coherence, 4),
            "order": self._stats.order_parameter,
            "conscious": self._stats.conscious,
            "phase": self._stats.phase_transition,
            "avalanche": self._stats.avalanche,
            "coupling": self._stats.soc_coupling,
            "lock_in": self._stats.lock_in,
            "symbol": self._stats.locked_symbol,
            "recognition": list(self._stats.last_recognition),
            "backend": self._stats.backend,
            "fps": round(self._stats.fps, 1),
        }

    def render(self) -> np.ndarray:
        cur = self._buffers[0]
        phase = cur[:, :, CH_PH]
        coh = coherence_map(cur)
        turing = np.abs(self._turing_u - self._turing_v)
        R = self._stats.order_parameter
        h = (phase / (2 * np.pi) + turing * 0.15) % 1.0
        s = np.clip(0.15 + coh * 0.7 + R * 0.2, 0, 1)
        v = np.clip(0.25 + coh * 0.45 + turing * 0.35 + (0.2 if self._stats.conscious else 0), 0, 1)
        rgb = _hsv_to_rgb(h, s, v)
        return (rgb * 255).astype(np.uint8)

    def render_composite(self, camera_bgr: np.ndarray | None = None) -> np.ndarray:
        brain = self.render()
        if camera_bgr is not None:
            import cv2

            th, tw = max(64, self.height // 5), max(80, self.width // 5)
            inset = cv2.resize(camera_bgr, (tw, th))
            brain_bgr = cv2.cvtColor(brain, cv2.COLOR_RGB2BGR)
            y0, x0 = 8, brain_bgr.shape[1] - tw - 8
            brain_bgr[y0 : y0 + th, x0 : x0 + tw] = inset
            cv2.rectangle(brain_bgr, (x0 - 1, y0 - 1), (x0 + tw, y0 + th), (40, 220, 180), 1)
            if self._stats.conscious:
                cv2.rectangle(brain_bgr, (2, 2), (brain_bgr.shape[1] - 3, brain_bgr.shape[0] - 3), (80, 180, 255), 2)
            return brain_bgr
        return brain[:, :, ::-1]

    def overlay_lines(self) -> list[str]:
        mind = "COSCIENTE" if self._stats.conscious else "pre-critico"
        lines = [
            f"{'CUDA' if self.uses_cuda else 'CPU'} tick={self._stats.tick} fps={self._stats.fps:.0f} · {mind}",
            f"R={self._stats.order_parameter:.3f} · {self._stats.phase_transition} · SOC={self._stats.soc_coupling:.3f}",
            f"valanga={self._stats.avalanche:.3f} · Turing={self._stats.turing_energy:.3f}",
        ]
        if self._stats.locked_symbol:
            lines.append(f"lock-in: {self._stats.locked_symbol} ({self._stats.lock_in:.2f})")
        for sym, sc in self._stats.last_recognition[:3]:
            if sym != self._stats.locked_symbol:
                lines.append(f"{sym}:{sc:.2f}")
        return lines

    def export_state_for_training(self) -> dict[str, np.ndarray]:
        cur = self._buffers[0]
        return {
            "impulse": cur[:, :, CH_IMP].copy(),
            "phase": cur[:, :, CH_PH].copy(),
            "weight": cur[:, :, CH_W].copy(),
            "turing_u": self._turing_u.copy(),
            "turing_v": self._turing_v.copy(),
            "order_parameter": np.array([self._stats.order_parameter], dtype=np.float32),
            "conscious": np.array([1.0 if self._stats.conscious else 0.0], dtype=np.float32),
            "tick": np.array([self._stats.tick], dtype=np.int32),
        }

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
