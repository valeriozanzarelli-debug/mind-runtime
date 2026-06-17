"""Test motore fisica V2."""

import numpy as np

from mindruntime.gpu_engine_v2 import BrainEngineV2
from mindruntime.gpu_physics_v2 import initialize_field_v2, physics_step_v2
from mindruntime.field_v2 import CH_IMP, CH_V, field_zeros, spike_times_zeros


def _spot(n=48):
    rgb = np.zeros((n, n, 3), dtype=np.float32)
    c = n // 2
    for y in range(n):
        for x in range(n):
            if (x - c) ** 2 + (y - c) ** 2 < (n // 5) ** 2:
                rgb[y, x] = 0.85
    return rgb


def test_initialize_v2_field():
    f = field_zeros(32, 32)
    initialize_field_v2(_spot(32), f, seed=1)
    assert f[:, :, CH_V].mean() < -50
    assert f[:, :, CH_IMP].max() > 0.1


def test_physics_step_v2():
    f = field_zeros(24, 24)
    s = field_zeros(24, 24)
    st = spike_times_zeros(24, 24)
    initialize_field_v2(_spot(24), f, seed=2)
    m = physics_step_v2(f, s, st, 1.0, do_soc=True)
    assert "order" in m
    assert m["order"] >= 0.0


def test_brain_engine_v2_loop():
    eng = BrainEngineV2(width=48, height=48)
    frame = (_spot(48) * 255).astype(np.uint8)
    for _ in range(8):
        r = eng.step(frame)
    assert r["tick"] == 8
    assert eng.export_state_for_training()["voltage"].shape == (48, 48)
    img = eng.render()
    assert img.shape == (48, 48, 3)
