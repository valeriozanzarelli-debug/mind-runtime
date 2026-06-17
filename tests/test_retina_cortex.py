"""Retina cortex + coscienza — neurone = pixel, sinapsi virtuali, lettura puntuale."""

import pytest

from organism.brain.consciousness_probe import ConsciousnessProbe
from organism.brain.retina_cortex import LongRangeSynapse, RetinaCortex


def _bright_spot_grid(size: int = 32, cx: int = 16, cy: int = 16, radius: int = 4) -> list[list[int]]:
    grid = [[0] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                grid[y][x] = 220
    return grid


def test_each_pixel_is_neuron():
    cortex = RetinaCortex(width=64, height=48)
    assert cortex.neuron_count == 64 * 48


def test_inject_pixels_stimulates_field():
    cortex = RetinaCortex(width=32, height=32)
    grid = _bright_spot_grid(32, cx=20, cy=10)
    fired = cortex.inject_pixels(grid)
    assert fired > 0
    cortex.propagate(steps=3)
    assert cortex.mean_activation() > 0.01


def test_virtual_synapses_spread_activation():
    cortex = RetinaCortex(width=32, height=32)
    cortex.inject_point(16, 16, intensity=0.95)
    before = cortex.active_ratio()
    cortex.propagate(steps=4)
    after = cortex.active_ratio()
    assert after >= before
    hotspots = cortex.hotspots(k=3)
    assert len(hotspots) >= 1
    # il centro o un vicino resta il punto più saliente
    top_x, top_y, _ = hotspots[0]
    assert abs(top_x - 16) <= 8 and abs(top_y - 16) <= 8


def test_long_range_synapse_links_distant_pixels():
    cortex = RetinaCortex(width=32, height=32)
    cortex.long_range.append(LongRangeSynapse(sy=4, sx=4, dy=28, dx=28, weight=0.4))
    cortex.inject_point(4, 4, intensity=1.0)
    cortex.propagate(steps=2)
    snap = ConsciousnessProbe().read(cortex)
    assert snap.active_neurons > 0
    distant = ConsciousnessProbe().read_at(cortex, 28, 28)
    assert distant.activation > 0.0 or snap.global_activation > 0.05


def test_consciousness_reads_precise_points():
    cortex = RetinaCortex(width=48, height=48)
    grid = _bright_spot_grid(48, cx=30, cy=12)
    cortex.inject_pixels(grid)
    cortex.propagate(steps=5)
    probe = ConsciousnessProbe(threshold=0.12)
    snap = probe.read(cortex, sensory_tags=["VIS:scene"], pressure=0.3)
    assert snap.focus is not None
    assert snap.focus.x == 30 or abs(snap.focus.x - 30) <= 6
    assert snap.conscious or snap.ignition > 0.1
    assert any("FOCUS:" in b for b in snap.broadcast)


def test_large_retina_scale():
    """256×256 = 65k neuroni — scala verso milioni."""
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        pytest.skip("numpy required for large retina")
    cortex = RetinaCortex(width=256, height=256)
    assert cortex.neuron_count == 65_536
    grid = _bright_spot_grid(256, cx=128, cy=128, radius=8)
    cortex.inject_pixels(grid)
    cortex.propagate(steps=2)
    assert cortex.stats()["neurons"] == 65_536
