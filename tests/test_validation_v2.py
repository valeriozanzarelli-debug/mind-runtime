"""Validation tests — criteri Prompt V2 Part 2."""

from __future__ import annotations

import numpy as np

from mindruntime.gpu_engine_v2 import BrainEngineV2
from mindruntime.gpu_physics_v2 import kuramoto_global


def _circle(n: int = 64) -> np.ndarray:
    img = np.zeros((n, n, 3), dtype=np.uint8)
    c = n // 2
    for y in range(n):
        for x in range(n):
            if (x - c) ** 2 + (y - c) ** 2 < (n // 5) ** 2:
                img[y, x] = 255
    return img


def _square(n: int = 64) -> np.ndarray:
    img = np.zeros((n, n, 3), dtype=np.uint8)
    m = n // 4
    img[m : n - m, m : n - m] = 255
    return img


def _noise(n: int = 48) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.random((n, n, 3)) * 255).astype(np.uint8)


def test_emergence_field_evolution():
    """Il campo evolve senza divergere; impulso e fase restano in range."""
    eng = BrainEngineV2(width=48, height=48, seed=1)
    frame = _noise(48)
    for _ in range(60):
        eng.step(frame)
    stats = eng.get_statistics()
    assert 0.0 <= stats["impulse_mean"] <= 1.0
    assert stats["impulse_std"] >= 0.0
    assert -90.0 <= stats["voltage_mean"] <= 40.0


def test_circle_vs_square_different_activation():
    eng_c = BrainEngineV2(width=48, height=48, seed=2)
    eng_s = BrainEngineV2(width=48, height=48, seed=2)
    for _ in range(30):
        eng_c.step(_circle(48))
        eng_s.step(_square(48))
    ac = eng_c.get_statistics()["active_neurons"]
    as_ = eng_s.get_statistics()["active_neurons"]
    assert ac > 0 and as_ > 0
    assert ac != as_ or eng_c.get_statistics()["impulse_mean"] != eng_s.get_statistics()["impulse_mean"]


def test_order_parameter_positive():
    eng = BrainEngineV2(width=48, height=48, seed=3)
    frame = _circle(48)
    for _ in range(30):
        eng.step(frame)
    R = eng.stats.order_parameter
    assert R > 0.0005


def test_free_energy_finite():
    eng = BrainEngineV2(width=32, height=32, seed=4)
    frame = _noise(32)
    for _ in range(20):
        eng.step(frame)
    fe = eng.stats.free_energy
    assert np.isfinite(fe)
    assert fe >= 0.0


def test_part2_api():
    eng = BrainEngineV2(width=32, height=32, seed=5)
    eng.step(_circle(32))
    for _ in range(5):
        eng.step(_circle(32))
    stats = eng.get_statistics()
    assert "active_neurons" in stats
    assert eng.step_count == 6
    zones = eng.get_recognition_zones(coherence_min=0.0)
    assert isinstance(zones, list)
    state = eng.export_state()
    assert state["field"].shape == (32, 32, 12)
    eng2 = BrainEngineV2(width=32, height=32, seed=99)
    eng2.import_state(state)
    assert eng2.step_count == 6
    for mode in ("phase_coherence", "voltage", "impulse"):
        img = eng.render(mode=mode)
        assert img.shape == (32, 32, 3)


def test_kuramoto_global_bounds():
    phase = np.random.default_rng(0).random((16, 16)).astype(np.float32) * 6.28
    R = kuramoto_global(phase)
    assert 0.0 <= R <= 1.0
