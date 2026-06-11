"""Motion motor — gesture primitives for avatar/robot (future)."""

from __future__ import annotations

from dataclasses import dataclass, field

from mind.types import MindResult


@dataclass
class MotionFrame:
    gesture: str
    intensity: float


@dataclass
class MotionResult:
    frames: list[MotionFrame]
    style: str
    symbols: list[str] = field(default_factory=list)


class MotionModule:
    ACTION_GESTURE = {
        "replace_bulb": "reach_up",
        "test_bulb": "inspect_close",
        "ask_placement": "open_palm",
        "ask_one_more": "tilt_head",
        "send_quote": "nod_confirm",
        "block_client": "palm_stop",
    }

    def __init__(self, brain, style: str = "neutral") -> None:
        self.brain = brain
        self.emitters = brain.get_neurons("motor", "motion_gesture_emitter")
        self.style = style

    def express(self, mind_result: MindResult) -> MotionResult:
        action_id = mind_result.action.id if mind_result.action else "idle"
        gesture = self.ACTION_GESTURE.get(action_id, "idle")
        intensity = 0.5 + (0.3 if mind_result.human_emotion_active else 0.0)
        frames = [MotionFrame(gesture=gesture, intensity=intensity)]
        for i, em in enumerate(self.emitters[:3]):
            em.fire(float(i), intensity=intensity)
        return MotionResult(
            frames=frames,
            style=self.style,
            symbols=[f"MOTOR:motion {gesture}@{intensity:.2f}"],
        )
