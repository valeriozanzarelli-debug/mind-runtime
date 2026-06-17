"""BrainEngineV2 — orchestrazione fisica suprema, API compatibile step/export."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mindruntime import cuda_util
from mindruntime.field_v2 import CH_COH, CH_EN, CH_IMP, CH_PH, CH_V, CH_W, TWO_PI, field_zeros, spike_times_zeros
from mindruntime.gpu_engine import _hsv_to_rgb, _resize_bilinear
from mindruntime.gpu_physics_v2 import (
    inject_rgb,
    initialize_field_v2,
    kuramoto_global,
    physics_step_v2,
)
from mindruntime.physics_core import build_phase_templates, phase_lock_scores
from mindruntime.resonators import TEMPLATE_NAMES, build_resonator_bank


@dataclass
class BrainV2Stats:
    tick: int = 0
    width: int = 0
    height: int = 0
    backend: str = "cpu"
    fps: float = 0.0
    order_parameter: float = 0.0
    mean_coherence: float = 0.0
    free_energy: float = 0.0
    conscious: bool = False
    phase_transition: str = "subcritical"
    locked_symbol: str = ""
    lock_in: float = 0.0
    last_recognition: list[tuple[str, float]] = field(default_factory=list)


class BrainEngineV2:
    """Motore V2 — HH RK4 + Turing + SOC + gamma + predictive coding."""

    def __init__(self, *, width: int = 256, height: int = 256, seed: int = 42) -> None:
        if width < 32 or height < 32:
            raise ValueError("risoluzione minima 32×32")
        self.width, self.height, self.seed = width, height, seed
        self._field = field_zeros(height, width)
        self._scratch = field_zeros(height, width)
        self._spike_time = spike_times_zeros(height, width)
        self._initialized = False
        self._stats = BrainV2Stats(width=width, height=height)
        bank = build_resonator_bank(TEMPLATE_NAMES)
        self._tpl_names = bank["names"]
        self._phase_tpl = build_phase_templates(bank["stack"])
        info = cuda_util.cuda_info()
        self._stats.backend = "cuda" if info.get("cuda") else "cpu"

    @property
    def stats(self) -> BrainV2Stats:
        return self._stats

    @property
    def uses_cuda(self) -> bool:
        return self._stats.backend == "cuda"

    def step(self, input_frame: np.ndarray | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        if input_frame is not None:
            rgb = self._resize_rgb(input_frame)
            if not self._initialized:
                initialize_field_v2(rgb, self._field, seed=self.seed)
                self._initialized = True
            else:
                inject_rgb(self._field, rgb)

        if not self._initialized:
            raise RuntimeError("chiama step() con un frame")

        self._stats.tick += 1
        metrics = physics_step_v2(
            self._field,
            self._scratch,
            self._spike_time,
            float(self._stats.tick),
            do_soc=self._stats.tick % 10 == 0,
        )

        scores = phase_lock_scores(self._field[:, :, CH_PH], self._phase_tpl)
        order = np.argsort(scores)[::-1]
        rec: list[tuple[str, float]] = []
        for idx in order[:5]:
            sc = float(scores[idx])
            if sc > 0.15:
                rec.append((self._tpl_names[int(idx)], sc))

        R = metrics["order"]
        self._stats.order_parameter = R
        self._stats.mean_coherence = metrics["mean_coherence"]
        self._stats.free_energy = metrics["free_energy"]
        self._stats.last_recognition = rec
        self._stats.lock_in = float(scores[order[0]]) if len(order) else 0.0
        self._stats.locked_symbol = rec[0][0] if rec and rec[0][1] > 0.35 else ""
        self._stats.phase_transition = (
            "supercritical" if R > 0.62 else ("critical" if R > 0.45 else "subcritical")
        )
        self._stats.conscious = R >= 0.52 and self._stats.mean_coherence > 0.25

        dt = time.perf_counter() - t0
        if dt > 0:
            self._stats.fps = 0.9 * self._stats.fps + 0.1 * (1.0 / dt)

        return {
            "tick": self._stats.tick,
            "order": round(R, 4),
            "conscious": self._stats.conscious,
            "phase": self._stats.phase_transition,
            "free_energy": round(self._stats.free_energy, 4),
            "recognition": rec,
            "backend": self._stats.backend,
            "fps": round(self._stats.fps, 1),
        }

    @property
    def step_count(self) -> int:
        return self._stats.tick

    def render(self, mode: str = "phase_coherence") -> np.ndarray:
        if mode == "phase_coherence":
            ph = self._field[:, :, CH_PH]
            coh = self._field[:, :, CH_COH]
            en = self._field[:, :, CH_EN]
            v = self._field[:, :, CH_V]
            vm = np.clip((v + 70) / 90, 0, 1)
            h = (ph / TWO_PI) % 1.0
            s = np.clip(0.2 + coh * 0.75, 0, 1)
            val = np.clip(0.25 + coh * 0.4 + en * 0.3 + vm * 0.2, 0, 1)
            rgb = _hsv_to_rgb(h, s, val)
            return (rgb * 255).astype(np.uint8)

        if mode == "voltage":
            import cv2

            voltage = self._field[:, :, CH_V]
            norm = np.clip((voltage + 80.0) / 120.0, 0, 1)
            gray = (norm * 255).astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if mode == "impulse":
            import cv2

            impulse = np.clip(self._field[:, :, CH_IMP] * 255, 0, 255).astype(np.uint8)
            return cv2.cvtColor(impulse, cv2.COLOR_GRAY2RGB)

        raise ValueError(f"Mode '{mode}' non riconosciuto")

    def render_composite(self, camera_bgr: np.ndarray | None = None) -> np.ndarray:
        brain = self.render()
        if camera_bgr is None:
            return brain[:, :, ::-1]
        import cv2

        th, tw = max(64, self.height // 5), max(80, self.width // 5)
        inset = cv2.resize(camera_bgr, (tw, th))
        bgr = cv2.cvtColor(brain, cv2.COLOR_RGB2BGR)
        y0, x0 = 8, bgr.shape[1] - tw - 8
        bgr[y0 : y0 + th, x0 : x0 + tw] = inset
        if self._stats.conscious:
            cv2.rectangle(bgr, (2, 2), (bgr.shape[1] - 3, bgr.shape[0] - 3), (90, 200, 255), 2)
        return bgr

    def overlay_lines(self) -> list[str]:
        mind = "COSCIENTE" if self._stats.conscious else "pre-critico"
        lines = [
            f"V2 {'CUDA' if self.uses_cuda else 'CPU'} tick={self._stats.tick} fps={self._stats.fps:.0f} · {mind}",
            f"R={self._stats.order_parameter:.3f} · FE={self._stats.free_energy:.3f} · {self._stats.phase_transition}",
        ]
        if self._stats.locked_symbol:
            lines.append(f"lock-in: {self._stats.locked_symbol} ({self._stats.lock_in:.2f})")
        for sym, sc in self._stats.last_recognition[:3]:
            if sym != self._stats.locked_symbol:
                lines.append(f"{sym}:{sc:.2f}")
        return lines

    def get_recognition_zones(self, coherence_min: float = 0.7) -> list[tuple[int, int, float]]:
        coh = self._field[:, :, CH_COH]
        active = np.argwhere(coh > coherence_min)
        zones = [(int(y), int(x), float(coh[y, x])) for y, x in active]
        zones.sort(key=lambda t: t[2], reverse=True)
        return zones

    def get_statistics(self) -> dict[str, float | int]:
        f = self._field
        return {
            "impulse_mean": float(f[:, :, CH_IMP].mean()),
            "impulse_std": float(f[:, :, CH_IMP].std()),
            "voltage_mean": float(f[:, :, CH_V].mean()),
            "coherence_mean": float(f[:, :, CH_COH].mean()),
            "coherence_max": float(f[:, :, CH_COH].max()),
            "active_neurons": int(np.sum(f[:, :, CH_IMP] > 0.5)),
            "spiking_neurons": int(np.sum(f[:, :, CH_V] > 0.0)),
            "order_parameter": self._stats.order_parameter,
            "free_energy": self._stats.free_energy,
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "field": self._field.copy(),
            "spike_time": self._spike_time.copy(),
            "shape": self._field.shape,
            "step_count": self._stats.tick,
            "timestamp": time.time(),
            "params": {
                "width": self.width,
                "height": self.height,
                "seed": self.seed,
                "order_parameter": self._stats.order_parameter,
                "mean_coherence": self._stats.mean_coherence,
            },
        }

    def import_state(self, state_dict: dict[str, Any]) -> None:
        field = state_dict.get("field")
        if field is None and "field_bytes" in state_dict:
            field = np.frombuffer(state_dict["field_bytes"], dtype=np.float32).reshape(state_dict["shape"])
        if field is None:
            raise ValueError("state_dict senza campo 'field'")
        field = np.asarray(field, dtype=np.float32)
        if field.shape != self._field.shape:
            raise ValueError(f"shape mismatch: {field.shape} vs {self._field.shape}")
        self._field[:] = field
        st = state_dict.get("spike_time")
        if st is not None:
            self._spike_time[:] = np.asarray(st, dtype=np.float32)
        self._stats.tick = int(state_dict.get("step_count", self._stats.tick))
        self._initialized = True

    def export_state_for_training(self) -> dict[str, np.ndarray]:
        return {
            "impulse": self._field[:, :, CH_IMP].copy(),
            "phase": self._field[:, :, CH_PH].copy(),
            "voltage": self._field[:, :, CH_V].copy(),
            "weight": self._field[:, :, CH_W].copy(),
            "coherence": self._field[:, :, CH_COH].copy(),
            "energy": self._field[:, :, CH_EN].copy(),
            "spike_time": self._spike_time.copy(),
            "order_parameter": np.array([self._stats.order_parameter], dtype=np.float32),
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


SupremeBrainEngine = BrainEngineV2
GPUBrainEngineV2 = BrainEngineV2
