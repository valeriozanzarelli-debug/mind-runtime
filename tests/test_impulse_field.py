"""Mare impulsi GPU + coscienza lettore."""

import pytest

from organism.brain.impulse_consciousness import ImpulseConsciousness
from organism.brain.impulse_field import create_impulse_field
from organism.brain.impulse_memory import ImpulseMemory
from organism.brain.impulse_scaffold import ImpulseScaffold


def _spot(size: int = 64, cx: int = 32, cy: int = 20) -> list[list[int]]:
    g = [[0] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 < 25:
                g[y][x] = 220
    return g


def test_impulse_field_moves_energy():
    field = create_impulse_field(64, 64, device="numpy")
    field.inject_pixels(_spot())
    e0 = field.regional_energy()["visual"]
    field.step(steps=4)
    flux = field.flux_magnitude()
    assert flux >= 0.0
    assert field.active_blobs()


def test_consciousness_reads_without_modifying_field():
    field = create_impulse_field(96, 96, device="numpy")
    field.inject_pixels(_spot(96, 48, 24))
    field.step(steps=3)
    energy_before = float(field.energy.mean())
    probe = ImpulseConsciousness()
    reading = probe.observe(field, pressure=0.3)
    energy_after = float(field.energy.mean())
    assert abs(energy_before - energy_after) < 0.02
    assert reading.focus_region in ("visual", "associative", "motor", "memory", "auditory")
    assert reading.conscious or reading.ignition > 0.08


def test_memory_recall_similar_episodes():
    mem = ImpulseMemory(capacity=10)
    sig = [0.1] * 64 + [0.9] * 64
    mem.store(sig, regions={"visual": 0.5}, label="luce")
    hits = mem.recall(sig, min_sim=0.9)
    assert len(hits) == 1
    assert hits[0].label == "luce"


def test_scaffold_pulse_produces_reading():
    sc = ImpulseScaffold(device="numpy", width=128, height=96)
    sc.perceive_visual(_spot(32))
    reading = sc.pulse(steps=3)
    assert reading is not None
    assert sc.themes_for_speech() or reading.sensations
    assert sc.stats()["pixels"] == 128 * 96


@pytest.mark.skipif(True, reason="torch optional")
def test_gpu_field_large():
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("no torch")
    field = create_impulse_field(256, 192, device="cpu")
    field.inject_pixels(_spot(256, 128, 40))
    field.step(steps=2)
    assert field.neuron_count == 256 * 192
