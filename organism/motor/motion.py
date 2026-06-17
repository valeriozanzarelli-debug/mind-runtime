"""Motion motor — gesti e orientamento corporeo virtuale (avatar/robot/spazio)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mind.types import MindResult


@dataclass
class MotionFrame:
    gesture: str
    intensity: float
    heading_deg: float = 0.0
    velocity: float = 0.0


@dataclass
class MotionResult:
    frames: list[MotionFrame]
    style: str
    symbols: list[str] = field(default_factory=list)
    body_mode: str = "idle"


class MotionModule:
    ACTION_GESTURE = {
        "replace_bulb": "reach_up",
        "test_bulb": "inspect_close",
        "ask_placement": "open_palm",
        "ask_one_more": "tilt_head",
        "send_quote": "nod_confirm",
        "block_client": "palm_stop",
    }

    SPATIAL_GESTURE = {
        "explore": "step_forward",
        "orient": "turn_toward",
        "freeze": "recoil",
        "drift": "idle_alert",
        "recoil": "recoil",
        "turn_toward": "turn_toward",
        "step_forward": "step_forward",
        "open_arms": "open_arms",
        "steady_self": "steady_self",
        "idle_alert": "idle_alert",
        "reach_up": "reach_up",
        "nod_confirm": "nod_confirm",
    }

    def __init__(self, brain, style: str = "neutral") -> None:
        self.brain = brain
        self.emitters = brain.get_neurons("motor", "motion_gesture_emitter")
        self.style = style

    def express(
        self,
        mind_result: MindResult | None = None,
        *,
        spatial_gesture: str = "",
        heading_deg: float = 0.0,
        velocity: float = 0.0,
        intensity: float | None = None,
        body_mode: str = "idle",
    ) -> MotionResult:
        if mind_result and mind_result.action:
            gesture = self.ACTION_GESTURE.get(mind_result.action.id, spatial_gesture or "idle")
            base_intensity = 0.5 + (0.3 if mind_result.human_emotion_active else 0.0)
        else:
            gesture = self.SPATIAL_GESTURE.get(spatial_gesture, spatial_gesture or "idle_alert")
            base_intensity = 0.45

        inten = intensity if intensity is not None else base_intensity
        inten = min(1.0, max(0.1, inten + velocity * 0.2))
        frames = [
            MotionFrame(
                gesture=gesture,
                intensity=inten,
                heading_deg=heading_deg,
                velocity=velocity,
            )
        ]
        for i, em in enumerate(self.emitters[:3]):
            em.fire(float(i), intensity=inten)
        return MotionResult(
            frames=frames,
            style=self.style,
            body_mode=body_mode,
            symbols=[
                f"MOTOR:motion {gesture}@{inten:.2f}",
                f"BODY:{body_mode} h={heading_deg:.0f}° v={velocity:.2f}",
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"style": self.style, "emitters": len(self.emitters)}
