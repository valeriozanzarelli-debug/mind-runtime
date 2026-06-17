"""Test fisica emergente — Turing, SOC, transizione di fase."""

import numpy as np

from mindruntime.dendritic_core import CH_IMP, N_CHANNELS, initialize_dendrites
from mindruntime.physics_core import (
    PhysicsState,
    build_phase_templates,
    kuramoto_order,
    physics_tick,
    turing_step,
)
from mindruntime.resonators import TEMPLATE_NAMES, build_resonator_bank


def _spot(n: int = 48) -> np.ndarray:
    rgb = np.zeros((n, n, 3), dtype=np.float32)
    c = n // 2
    for y in range(n):
        for x in range(n):
            if (x - c) ** 2 + (y - c) ** 2 < (n // 5) ** 2:
                rgb[y, x] = 0.9
    return rgb


def test_kuramoto_order_bounded():
    ph = np.random.rand(32, 32).astype(np.float32) * 6.28
    R = kuramoto_order(ph)
    assert 0.0 <= R <= 1.0


def test_turing_produces_pattern():
    u = np.ones((40, 40), dtype=np.float32) * 0.5
    v = np.ones((40, 40), dtype=np.float32) * 0.25
    u[18:22, 18:22] = 0.8
    e0 = float(np.std(u))
    for _ in range(30):
        u, v = turing_step(u, v)
    assert float(np.std(u)) > e0 * 0.5


def test_physics_tick_returns_consciousness_fields():
    h, w = 48, 48
    state = np.zeros((h, w, N_CHANNELS), dtype=np.float32)
    initialize_dendrites(_spot(h), state, seed=3)
    u = np.ones((h, w), dtype=np.float32) * 0.5
    v = np.ones((h, w), dtype=np.float32) * 0.25
    bank = build_resonator_bank(TEMPLATE_NAMES[:4], size=24)
    tpl_ph = build_phase_templates(bank["stack"])
    phys = PhysicsState()
    prev = state[:, :, CH_IMP].copy()
    out = physics_tick(state, prev, u, v, phys, tpl_ph, bank["names"])
    assert "order" in out
    assert "conscious" in out
    assert "recognition" in out
    assert phys.order_parameter >= 0.0


def test_engine_reports_order_parameter():
    from mindruntime.dendritic_engine import DendriticBrainEngine

    eng = DendriticBrainEngine(width=48, height=48)
    frame = (_spot(48) * 255).astype(np.uint8)
    result = eng.step(frame)
    for _ in range(15):
        result = eng.step(frame)
    assert "order" in result
    assert eng.stats.order_parameter >= 0.0
    assert eng.stats.phase_transition in ("subcritical", "critical", "supercritical")
