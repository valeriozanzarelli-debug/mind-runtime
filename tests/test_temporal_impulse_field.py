"""Test campo impulsi temporale — fase, triple-buffer, gravità."""

import os

import pytest

from organism.brain.impulse_consciousness import ImpulseConsciousness
from organism.brain.impulse_field import create_impulse_field
from organism.brain.impulse_scaffold import ImpulseScaffold
from organism.brain.resonance_templates import build_template_bank, correlate_template
from organism.brain.temporal_impulse_field import create_temporal_impulse_field


def _spot(size: int = 64, cx: int = 32, cy: int = 20) -> list[list[int]]:
    g = [[0] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 < 25:
                g[y][x] = 220
    return g


def test_factory_uses_temporal_by_default():
    os.environ["ORGANISM_TEMPORAL"] = "1"
    field = create_impulse_field(64, 64, device="numpy")
    assert getattr(field, "temporal", False) is True


def test_temporal_triple_buffer_and_phase():
    field = create_temporal_impulse_field(96, 96, device="numpy")
    field.inject_pixels(_spot(96, 48, 24))
    field.step(steps=8)
    assert field.tick == 8.0
    assert field.phase_coherence() >= 0.0
    assert field.acceleration_magnitude() >= 0.0
    assert len(field.to_phase_bytes()) == 96 * 96


def test_temporal_mass_grows_with_activity():
    field = create_temporal_impulse_field(64, 64, device="numpy")
    m0 = field.stats()["mean_mass"]
    for _ in range(12):
        field.inject_pixels(_spot(64, 32, 16))
        field.step(steps=2)
    m1 = field.stats()["mean_mass"]
    assert m1 >= m0


def test_resonance_templates_bank():
    bank = build_template_bank()
    assert len(bank["symbols"]) == 36
    a = bank["templates"]["A"]
    b = bank["templates"]["B"]
    assert correlate_template(a, a) > 0.99
    assert correlate_template(a, b) < correlate_template(a, a)


def test_consciousness_reads_temporal_field():
    field = create_temporal_impulse_field(96, 96, device="numpy")
    field.inject_pixels(_spot(96, 48, 24))
    field.step(steps=6)
    probe = ImpulseConsciousness()
    reading = probe.observe(field, pressure=0.25)
    assert reading.phase_coherence >= 0.0
    assert reading.acceleration >= 0.0


def test_scaffold_temporal_pulse():
    os.environ["ORGANISM_TEMPORAL"] = "1"
    sc = ImpulseScaffold(device="numpy", width=128, height=96)
    assert sc.field.temporal is True
    sc.perceive_visual(_spot(32))
    reading = sc.pulse(steps=6)
    assert reading is not None
    stats = sc.stats()
    assert stats.get("temporal") is True
    assert sc.phase_bytes() is not None


def test_legacy_frame_field_when_disabled():
    os.environ["ORGANISM_TEMPORAL"] = "0"
    field = create_impulse_field(64, 64, device="numpy")
    assert getattr(field, "temporal", False) is False
    os.environ["ORGANISM_TEMPORAL"] = "1"
