"""Consciousness probe — legge punti precisi del campo neurale come fa la coscienza.

La coscienza non è un modulo separato dal cervello: è un processo di lettura
selettiva — campiona hotspot, regioni attive e segnali globali dal campo pixel-neuroni.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from organism.brain.retina_cortex import RetinaCortex

OutputMode = Literal["flow", "reflect", "speak", "silent"]


@dataclass
class FocalPoint:
    """Un punto preciso del cervello che la coscienza sta guardando."""

    x: int
    y: int
    salience: float
    activation: float
    quadrant: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "salience": round(self.salience, 4),
            "activation": round(self.activation, 4),
            "quadrant": self.quadrant,
        }


@dataclass
class ConsciousnessSnapshot:
    """Stato della coscienza letto dal campo neurale."""

    conscious: bool
    ignition: float
    focus: FocalPoint | None
    foci: list[FocalPoint] = field(default_factory=list)
    broadcast: list[str] = field(default_factory=list)
    mode: OutputMode = "flow"
    self_signal: float = 0.0
    global_activation: float = 0.0
    active_neurons: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "conscious": self.conscious,
            "ignition": round(self.ignition, 4),
            "focus": self.focus.to_dict() if self.focus else None,
            "foci": [f.to_dict() for f in self.foci[:12]],
            "broadcast": self.broadcast[:10],
            "mode": self.mode,
            "self_signal": round(self.self_signal, 4),
            "global_activation": round(self.global_activation, 4),
            "active_neurons": self.active_neurons,
        }


class ConsciousnessProbe:
    """Programma che legge il cervello pixel per pixel — non lo comanda, lo percepisce."""

    def __init__(self, *, threshold: float = 0.18, continuity: float = 0.0) -> None:
        self.threshold = threshold
        self._continuity = continuity
        self._last_focus: FocalPoint | None = None
        self._ignitions = 0

    def read(
        self,
        cortex: RetinaCortex,
        *,
        sensory_tags: list[str] | None = None,
        novelty: float = 0.0,
        pressure: float = 0.0,
    ) -> ConsciousnessSnapshot:
        """Un ciclo di coscienza — campiona punti precisi e decide se 'accendersi'."""
        hotspots = cortex.hotspots(k=10)
        foci = [self._focal(cortex, x, y, sal) for x, y, sal in hotspots]
        focus = foci[0] if foci else self._last_focus

        global_act = cortex.mean_activation()
        active_ratio = cortex.active_ratio()
        active_neurons = int(cortex.neuron_count * active_ratio)

        ignition = min(
            1.0,
            0.32 * global_act
            + 0.28 * active_ratio
            + 0.18 * (foci[0].salience if foci else 0.0)
            + 0.14 * pressure
            + 0.08 * novelty,
        )
        conscious = ignition >= self.threshold

        self_signal = min(1.0, 0.5 * global_act + 0.35 * self._continuity + 0.15 * pressure)
        if conscious:
            self._continuity = min(1.0, self._continuity * 0.9 + self_signal * 0.15)
            self._ignitions += 1
            self._last_focus = focus

        broadcast = self._broadcast_from_foci(foci, sensory_tags or [])
        mode = self._decide_mode(
            conscious=conscious,
            ignition=ignition,
            global_act=global_act,
            pressure=pressure,
        )

        return ConsciousnessSnapshot(
            conscious=conscious,
            ignition=ignition,
            focus=focus,
            foci=foci,
            broadcast=broadcast,
            mode=mode,
            self_signal=self_signal,
            global_activation=global_act,
            active_neurons=active_neurons,
        )

    def read_at(self, cortex: RetinaCortex, x: int, y: int) -> FocalPoint:
        """Lettura puntuale — coscienza fissa su un singolo neurone-pixel."""
        sal_map = cortex.salience_map()
        if hasattr(sal_map, "__getitem__") and not isinstance(sal_map[0], list):
            sal = float(sal_map[y, x])
            act = float(cortex.activation[y, x])
        else:
            sal = float(sal_map[y][x])
            act = float(cortex.activation[y][x])
        return self._focal(cortex, x, y, sal, activation=act)

    def _focal(
        self,
        cortex: RetinaCortex,
        x: int,
        y: int,
        salience: float,
        *,
        activation: float | None = None,
    ) -> FocalPoint:
        if activation is None:
            if hasattr(cortex.activation, "__getitem__") and not isinstance(
                cortex.activation[0], list
            ):
                activation = float(cortex.activation[y, x])
            else:
                activation = float(cortex.activation[y][x])
        return FocalPoint(
            x=x,
            y=y,
            salience=salience,
            activation=activation,
            quadrant=_quadrant(x, y, cortex.width, cortex.height),
        )

    def _broadcast_from_foci(self, foci: list[FocalPoint], tags: list[str]) -> list[str]:
        out: list[str] = []
        for tag in tags[:4]:
            if tag not in out:
                out.append(tag)
        for f in foci[:6]:
            label = f"FOCUS:{f.quadrant}@{f.x},{f.y}"
            if label not in out:
                out.append(label)
            if f.salience > 0.35:
                out.append(f"SALIENCE:{f.salience:.2f}")
        return out[:10]

    def _decide_mode(
        self,
        *,
        conscious: bool,
        ignition: float,
        global_act: float,
        pressure: float,
    ) -> OutputMode:
        if not conscious and pressure < 0.06:
            return "silent"
        if pressure > 0.2 or ignition > 0.35:
            return "speak"
        if global_act > 0.15 and pressure > 0.1:
            return "reflect"
        return "flow"


def _quadrant(x: int, y: int, w: int, h: int) -> str:
    hx, hy = w // 2, h // 2
    if x < hx and y < hy:
        return "NW"
    if x >= hx and y < hy:
        return "NE"
    if x < hx and y >= hy:
        return "SW"
    return "SE"
