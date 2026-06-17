"""Schema corporeo virtuale — orientamento, vestibolare, propriocezione, ippocampo.

Senza corpo fisico: il cervello mantiene coordinate egocentriche, traiettorie
e place cells per muoversi e orientarsi nello spazio percepito (flusso ottico,
salienza visiva, impulso esplorativo).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpatialPose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    velocity: float = 0.0
    angular_velocity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "heading_deg": round(math.degrees(self.heading), 1),
            "pitch_deg": round(math.degrees(self.pitch), 1),
            "roll_deg": round(math.degrees(self.roll), 1),
            "velocity": round(self.velocity, 3),
            "angular_velocity": round(self.angular_velocity, 3),
        }


@dataclass
class PlaceCell:
    cx: float
    cy: float
    activation: float = 0.0
    visits: int = 0


@dataclass
class VestibularState:
    balance: float = 0.85
    vertigo: float = 0.0
    grav_vector: tuple[float, float, float] = (0.0, -1.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "balance": round(self.balance, 3),
            "vertigo": round(self.vertigo, 3),
            "grav": [round(v, 3) for v in self.grav_vector],
        }


@dataclass
class VirtualBodySchema:
    """Corpo astratto — posture, arti virtuali, navigazione."""

    pose: SpatialPose = field(default_factory=SpatialPose)
    vestibular: VestibularState = field(default_factory=VestibularState)
    proprioception: dict[str, float] = field(
        default_factory=lambda: {
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "torso_lean": 0.0,
            "arm_reach": 0.0,
            "stance_width": 0.5,
        }
    )
    _place_cells: list[PlaceCell] = field(default_factory=list)
    _grid_size: float = 2.0
    _pulse: int = 0
    navigate_mode: str = "explore"

    def tick(
        self,
        *,
        optical_flow: float = 0.0,
        flow_direction: float = 0.0,
        salient_bearing: float | None = None,
        curiosity: float = 0.5,
        fear: float = 0.1,
        motion_gesture: str = "",
    ) -> SpatialPose:
        self._pulse += 1
        p = self.pose

        # Vestibolare — flusso ottico + rotazione simulata
        spin = abs(p.angular_velocity) + optical_flow * 0.3
        self.vestibular.vertigo = _clamp(spin * 0.4)
        self.vestibular.balance = _clamp(0.9 - self.vestibular.vertigo - fear * 0.25)

        if fear > 0.55:
            self.navigate_mode = "freeze"
        elif curiosity > 0.6 and optical_flow < 0.08:
            self.navigate_mode = "explore"
        elif salient_bearing is not None:
            self.navigate_mode = "orient"
        else:
            self.navigate_mode = "drift"

        dt = 0.05
        if self.navigate_mode == "explore":
            p.heading += (curiosity - 0.4) * 0.08
            p.velocity = _clamp(0.15 + curiosity * 0.35)
        elif self.navigate_mode == "orient" and salient_bearing is not None:
            delta = _angle_diff(salient_bearing, p.heading)
            p.angular_velocity = _clamp(delta * 2.0, -1.5, 1.5)
            p.heading += p.angular_velocity * dt
            p.velocity = _clamp(abs(delta) * 0.2)
        elif self.navigate_mode == "freeze":
            p.velocity *= 0.85
            p.angular_velocity *= 0.7
        else:
            p.velocity = _clamp(optical_flow * 0.5 + 0.05)
            p.heading += flow_direction * optical_flow * 0.15

        p.x += math.cos(p.heading) * p.velocity * dt
        p.y += math.sin(p.heading) * p.velocity * dt
        p.pitch = _clamp(p.pitch * 0.9 + optical_flow * 0.05 - 0.02, -0.4, 0.4)
        p.roll = _clamp(p.roll * 0.92 + p.angular_velocity * 0.08, -0.35, 0.35)

        self._update_place_cells(p.x, p.y, novelty=optical_flow)
        self._update_proprioception(motion_gesture, optical_flow)
        return p

    def salient_bearing_from_features(self, features: dict[str, Any]) -> float | None:
        """Stima direzione egocentrica da blob/salienza visiva."""
        gist = features.get("gist") or {}
        patch = gist.get("salient_patch")
        if isinstance(patch, (list, tuple)) and len(patch) >= 2:
            nx = float(patch[0]) - 0.5
            ny = float(patch[1]) - 0.5
            if abs(nx) + abs(ny) > 0.08:
                return math.atan2(ny, nx)
        motion = float(features.get("motion", 0.0))
        if motion > 0.12:
            return self.pose.heading + 0.2
        return None

    def gesture_from_state(self, *, joy: float = 0.2, fear: float = 0.1) -> str:
        if fear > 0.55:
            return "recoil"
        if self.navigate_mode == "orient":
            return "turn_toward"
        if self.pose.velocity > 0.25:
            return "step_forward"
        if joy > 0.55:
            return "open_arms"
        if self.vestibular.vertigo > 0.4:
            return "steady_self"
        return "idle_alert"

    def hippocampus_context(self, limit: int = 4) -> list[str]:
        active = sorted(self._place_cells, key=lambda c: -c.activation)[:limit]
        return [f"luogo({round(c.cx,1)},{round(c.cy,1)})" for c in active if c.activation > 0.15]

    def stats(self) -> dict[str, Any]:
        return {
            "pose": self.pose.to_dict(),
            "vestibular": self.vestibular.to_dict(),
            "proprioception": {k: round(v, 3) for k, v in self.proprioception.items()},
            "navigate_mode": self.navigate_mode,
            "place_cells_active": sum(1 for c in self._place_cells if c.activation > 0.2),
            "pulse": self._pulse,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.stats(),
            "place_cells": [
                {"cx": c.cx, "cy": c.cy, "act": round(c.activation, 3), "visits": c.visits}
                for c in self._place_cells[-24:]
            ],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        pose = data.get("pose", {})
        p = self.pose
        p.x = float(pose.get("x", p.x))
        p.y = float(pose.get("y", p.y))
        p.z = float(pose.get("z", p.z))
        p.heading = math.radians(float(pose.get("heading_deg", math.degrees(p.heading))))
        p.velocity = float(pose.get("velocity", p.velocity))
        vest = data.get("vestibular", {})
        self.vestibular.balance = float(vest.get("balance", self.vestibular.balance))
        self.vestibular.vertigo = float(vest.get("vertigo", self.vestibular.vertigo))
        prop = data.get("proprioception", {})
        for k, v in prop.items():
            self.proprioception[k] = float(v)
        self.navigate_mode = str(data.get("navigate_mode", self.navigate_mode))
        self._pulse = int(data.get("pulse", 0))
        cells = data.get("place_cells", [])
        self._place_cells = [
            PlaceCell(cx=float(c["cx"]), cy=float(c["cy"]), activation=float(c.get("act", 0)), visits=int(c.get("visits", 0)))
            for c in cells
            if "cx" in c and "cy" in c
        ]

    def _update_place_cells(self, x: float, y: float, *, novelty: float) -> None:
        if not self._place_cells:
            for i in range(12):
                ang = i / 12 * 2 * math.pi
                self._place_cells.append(
                    PlaceCell(cx=math.cos(ang) * 3, cy=math.sin(ang) * 3)
                )
        sigma = self._grid_size
        for cell in self._place_cells:
            dist = math.hypot(x - cell.cx, y - cell.cy)
            field = math.exp(-(dist * dist) / (2 * sigma * sigma))
            cell.activation = _clamp(cell.activation * 0.85 + field * (0.35 + novelty * 0.2))
            if field > 0.55:
                cell.visits += 1
                cell.cx = 0.92 * cell.cx + 0.08 * x

    def _update_proprioception(self, gesture: str, flow: float) -> None:
        pr = self.proprioception
        pr["head_yaw"] = _clamp(pr.get("head_yaw", 0) * 0.85 + self.pose.angular_velocity * 0.4)
        pr["head_pitch"] = _clamp(pr.get("head_pitch", 0) * 0.9 + self.pose.pitch)
        pr["torso_lean"] = _clamp(pr.get("torso_lean", 0) * 0.88 + self.pose.roll * 0.5)
        if gesture == "reach_up":
            pr["arm_reach"] = _clamp(pr.get("arm_reach", 0) + 0.25)
        else:
            pr["arm_reach"] = _clamp(pr.get("arm_reach", 0) * 0.9)
        pr["stance_width"] = _clamp(0.45 + flow * 0.2 + self.vestibular.vertigo * 0.15)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _angle_diff(target: float, current: float) -> float:
    d = target - current
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d
