"""Test schema corporeo virtuale — orientamento e place cells."""

from organism.cognition.spatial_body import VirtualBodySchema


def test_explore_increases_velocity():
    body = VirtualBodySchema()
    body.tick(curiosity=0.85, optical_flow=0.02)
    assert body.pose.velocity > 0.1
    assert body.navigate_mode in ("explore", "drift", "orient")


def test_fear_freezes():
    body = VirtualBodySchema()
    body.tick(fear=0.8, curiosity=0.9)
    assert body.navigate_mode == "freeze"


def test_orient_toward_salient_bearing():
    body = VirtualBodySchema()
    body.tick(salient_bearing=1.2, curiosity=0.6)
    assert body.navigate_mode == "orient"


def test_hippocampus_context_after_movement():
    body = VirtualBodySchema()
    for _ in range(8):
        body.tick(optical_flow=0.2, curiosity=0.7)
    ctx = body.hippocampus_context()
    assert isinstance(ctx, list)


def test_gesture_from_state():
    body = VirtualBodySchema()
    assert body.gesture_from_state(fear=0.7) == "recoil"
    assert body.gesture_from_state(joy=0.8) in ("open_arms", "idle_alert", "step_forward")
