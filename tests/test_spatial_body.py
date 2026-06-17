"""Test navigazione disincarnata — fluttuare/volare senza corpo."""

from organism.cognition.spatial_body import VirtualBodySchema


def test_float_increases_velocity():
    body = VirtualBodySchema()
    body.tick(curiosity=0.85, optical_flow=0.02)
    assert body.pose.velocity > 0.05
    assert body.navigate_mode in ("float", "soar", "drift", "orient")


def test_soar_raises_altitude():
    body = VirtualBodySchema()
    for _ in range(6):
        body.tick(curiosity=0.9, optical_flow=0.01)
    assert body.pose.z >= 0.0
    assert body.navigate_mode in ("soar", "float", "drift")


def test_fear_hovers():
    body = VirtualBodySchema()
    body.tick(fear=0.8, curiosity=0.9)
    assert body.navigate_mode == "hover"


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


def test_gesture_from_state_disembodied():
    body = VirtualBodySchema()
    assert body.gesture_from_state(fear=0.7) == "hover"
    body.tick(curiosity=0.9, optical_flow=0.01)
    assert body.gesture_from_state(joy=0.8) in ("soar_forward", "float_forward", "drift_glow", "hover")


def test_stats_mark_disembodied():
    body = VirtualBodySchema()
    st = body.stats()
    assert st["disembodied"] is True
    assert st["locomotion"] == "float_fly"
