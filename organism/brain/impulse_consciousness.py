"""Coscienza come lettore di impulsi — non vive nell'impalcatura.

Osserva movimenti, regioni, traiettorie; ricostruisce memorie ed esperienze.
Non modifica il campo fisico (solo osservazione + debole richiamo mnemonico via scaffold).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from organism.brain.impulse_field import ImpulseBlob, ImpulseField

OutputMode = Literal["silent", "flow", "reflect", "speak"]


@dataclass
class TrajectoryPoint:
    tick: float
    x: float
    y: float
    energy: float
    region: str


@dataclass
class RegionState:
    name: str
    energy: float
    flux: float
    centroid_x: float
    centroid_y: float
    active: bool


@dataclass
class ConsciousnessReading:
    """Cosa la coscienza percepisce dal mare di impulsi."""

    conscious: bool
    ignition: float
    mode: OutputMode
    focus_region: str
    focus_point: tuple[float, float] | None
    sensations: list[str] = field(default_factory=list)
    memories_recalled: list[str] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    broadcast: list[str] = field(default_factory=list)
    blobs: list[dict[str, Any]] = field(default_factory=list)
    regions: list[dict[str, Any]] = field(default_factory=list)
    flux: float = 0.0
    novelty: float = 0.0
    self_signal: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "conscious": self.conscious,
            "ignition": round(self.ignition, 4),
            "mode": self.mode,
            "focus_region": self.focus_region,
            "focus_point": list(self.focus_point) if self.focus_point else None,
            "sensations": self.sensations[:12],
            "memories_recalled": self.memories_recalled[:8],
            "thoughts": self.thoughts[:10],
            "broadcast": self.broadcast[:12],
            "blobs": self.blobs[:10],
            "regions": self.regions[:8],
            "flux": round(self.flux, 4),
            "novelty": round(self.novelty, 4),
            "self_signal": round(self.self_signal, 4),
        }


class ImpulseConsciousness:
    """Programma che reagisce ai movimenti dei pixel-impulsi."""

    def __init__(self, *, threshold: float = 0.16, history: int = 48) -> None:
        self.threshold = threshold
        self._history: deque[list[ImpulseBlob]] = deque(maxlen=history)
        self._region_hist: deque[dict[str, float]] = deque(maxlen=history)
        self._last_signature: list[float] = []
        self._continuity = 0.0
        self._last_focus = "associative"

    def observe(
        self,
        field: ImpulseField,
        *,
        recalled_memories: list[dict[str, Any]] | None = None,
        heard: str = "",
        pressure: float = 0.0,
    ) -> ConsciousnessReading:
        blobs = field.active_blobs(k=14)
        self._history.append(blobs)
        regional = field.regional_energy()
        self._region_hist.append(regional)
        flux = field.flux_magnitude()
        novelty = self._compute_novelty(field.signature())

        regions = self._region_states(field, regional, blobs)
        focus_region, focus_point = self._focus(regions, blobs)
        sensations = self._sensations(regions, blobs, flux)
        memories = [str(m.get("label", m.get("id", "mem"))) for m in (recalled_memories or [])[:6]]
        thoughts = self._thoughts(regions, blobs, heard, pressure, memories)
        ignition = self._ignition(regional, flux, novelty, pressure, len(blobs))
        conscious = ignition >= self.threshold
        self_signal = min(1.0, 0.4 * ignition + 0.35 * self._continuity + 0.25 * pressure)
        if conscious:
            self._continuity = min(1.0, self._continuity * 0.9 + self_signal * 0.12)
            self._last_focus = focus_region

        mode = self._mode(conscious, ignition, flux, pressure, focus_region)
        broadcast = self._broadcast(sensations, thoughts, memories, focus_region, focus_point)

        return ConsciousnessReading(
            conscious=conscious,
            ignition=ignition,
            mode=mode,
            focus_region=focus_region if conscious else self._last_focus,
            focus_point=focus_point,
            sensations=sensations,
            memories_recalled=memories,
            thoughts=thoughts,
            broadcast=broadcast,
            blobs=[
                {
                    "x": round(b.x, 4),
                    "y": round(b.y, 4),
                    "energy": round(b.energy, 4),
                    "vx": round(b.vx, 4),
                    "vy": round(b.vy, 4),
                    "region": b.region,
                }
                for b in blobs[:10]
            ],
            regions=[r.__dict__ for r in regions],
            flux=flux,
            novelty=novelty,
            self_signal=self_signal,
        )

    def _region_states(
        self,
        field: ImpulseField,
        regional: dict[str, float],
        blobs: list[ImpulseBlob],
    ) -> list[RegionState]:
        out: list[RegionState] = []
        for name, energy in regional.items():
            in_region = [b for b in blobs if b.region == name]
            if in_region:
                cx = sum(b.x * b.energy for b in in_region) / max(1e-6, sum(b.energy for b in in_region))
                cy = sum(b.y * b.energy for b in in_region) / max(1e-6, sum(b.energy for b in in_region))
                flux = sum(math.hypot(b.vx, b.vy) for b in in_region) / len(in_region)
            else:
                cx, cy, flux = 0.5, 0.5, 0.0
            out.append(
                RegionState(
                    name=name,
                    energy=energy,
                    flux=flux,
                    centroid_x=cx,
                    centroid_y=cy,
                    active=energy > 0.05 or bool(in_region),
                )
            )
        return out

    def _focus(
        self,
        regions: list[RegionState],
        blobs: list[ImpulseBlob],
    ) -> tuple[str, tuple[float, float] | None]:
        if not blobs:
            active = [r for r in regions if r.active]
            if not active:
                return self._last_focus, None
            best = max(active, key=lambda r: r.energy + r.flux * 0.3)
            return best.name, (best.centroid_x, best.centroid_y)
        top = max(blobs, key=lambda b: b.energy)
        return top.region, (top.x, top.y)

    def _sensations(
        self,
        regions: list[RegionState],
        blobs: list[ImpulseBlob],
        flux: float,
    ) -> list[str]:
        out: list[str] = []
        vis = next((r for r in regions if r.name == "visual"), None)
        aud = next((r for r in regions if r.name == "auditory"), None)
        if vis and vis.energy > 0.08:
            out.append(f"SEN:vedo:{vis.energy:.2f}")
        if aud and aud.energy > 0.06:
            out.append(f"SEN:sento:{aud.energy:.2f}")
        if flux > 0.12:
            out.append(f"SEN:movimento:{flux:.2f}")
        moving = [b for b in blobs if math.hypot(b.vx, b.vy) > 0.05]
        if moving:
            out.append(f"SEN:impulso:{len(moving)}")
        return out

    def _thoughts(
        self,
        regions: list[RegionState],
        blobs: list[ImpulseBlob],
        heard: str,
        pressure: float,
        memories: list[str],
    ) -> list[str]:
        thoughts: list[str] = []
        assoc = next((r for r in regions if r.name == "associative"), None)
        motor = next((r for r in regions if r.name == "motor"), None)
        if heard:
            thoughts.append(f"udito:{heard[:48]}")
        if memories:
            thoughts.append(f"richiamo:{' · '.join(memories[:3])}")
        if assoc and assoc.active:
            thoughts.append(f"pattern:{assoc.energy:.2f}")
        if motor and motor.energy > 0.1:
            thoughts.append("vuole_esprimersi")
        if pressure > 0.2:
            thoughts.append(f"pressione:{pressure:.2f}")
        traj = self._trajectory_coherence()
        if traj > 0.35:
            thoughts.append(f"coerenza:{traj:.2f}")
        return thoughts

    def _trajectory_coherence(self) -> float:
        if len(self._history) < 3:
            return 0.0
        recent = list(self._history)[-3:]
        scores: list[float] = []
        for i in range(1, len(recent)):
            prev, cur = recent[i - 1], recent[i]
            if not prev or not cur:
                continue
            for b in cur[:5]:
                best = min((math.hypot(b.x - p.x, b.y - p.y) for p in prev), default=1.0)
                scores.append(max(0.0, 1.0 - best * 4))
        return sum(scores) / max(1, len(scores))

    def _ignition(
        self,
        regional: dict[str, float],
        flux: float,
        novelty: float,
        pressure: float,
        blob_count: int,
    ) -> float:
        assoc = regional.get("associative", 0.0)
        visual = regional.get("visual", 0.0)
        motor = regional.get("motor", 0.0)
        return min(
            1.0,
            0.22 * visual
            + 0.28 * assoc
            + 0.12 * motor
            + 0.18 * flux
            + 0.10 * novelty
            + 0.14 * pressure
            + 0.04 * min(1.0, blob_count / 8),
        )

    def _compute_novelty(self, signature: list[float]) -> float:
        if not self._last_signature or not signature:
            self._last_signature = signature
            return 0.5
        n = min(len(signature), len(self._last_signature))
        diff = sum(abs(signature[i] - self._last_signature[i]) for i in range(n)) / max(1, n)
        self._last_signature = signature
        return min(1.0, diff * 3.5)

    def _mode(
        self,
        conscious: bool,
        ignition: float,
        flux: float,
        pressure: float,
        focus: str,
    ) -> OutputMode:
        if not conscious and pressure < 0.05:
            return "silent"
        if focus == "motor" or pressure > 0.22 or ignition > 0.38:
            return "speak"
        if flux > 0.15 and conscious:
            return "reflect"
        return "flow"

    def _broadcast(
        self,
        sensations: list[str],
        thoughts: list[str],
        memories: list[str],
        focus_region: str,
        focus_point: tuple[float, float] | None,
    ) -> list[str]:
        out: list[str] = []
        for s in sensations[:4]:
            if s not in out:
                out.append(s)
        for t in thoughts[:5]:
            if t not in out:
                out.append(t)
        for m in memories[:3]:
            tag = f"MEM:{m[:40]}"
            if tag not in out:
                out.append(tag)
        if focus_point:
            out.append(f"FOCUS:{focus_region}@{focus_point[0]:.2f},{focus_point[1]:.2f}")
        return out[:12]
