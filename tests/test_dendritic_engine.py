"""Test compartimenti dendritici + engine nativo."""

import numpy as np

from mindruntime.dendritic_core import (
    CH_CA,
    CH_IMP,
    CH_NA,
    N_CHANNELS,
    backward_dendrite,
    coherence_map,
    forward_dendrite,
    initialize_dendrites,
)
from mindruntime.dendritic_engine import DendriticBrainEngine


def _rgb_spot(size: int = 64) -> np.ndarray:
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    c = size // 2
    for y in range(size):
        for x in range(size):
            if (x - c) ** 2 + (y - c) ** 2 < (size // 5) ** 2:
                rgb[y, x] = 0.9
    return rgb


def test_dendrite_init_has_ion_channels():
    h, w = 48, 48
    state = np.zeros((h, w, N_CHANNELS), dtype=np.float32)
    initialize_dendrites(_rgb_spot(h), state, seed=1)
    assert state[:, :, CH_NA].mean() > 0.05
    assert state[:, :, CH_CA].mean() > 0.0
    assert state[:, :, CH_IMP].max() > 0.1


def test_forward_and_backward_step():
    h, w = 40, 40
    s = np.zeros((h, w, N_CHANNELS), dtype=np.float32)
    initialize_dendrites(_rgb_spot(h), s, seed=2)
    t1, t2 = s.copy(), s.copy()
    out = np.zeros_like(s)
    forward_dendrite(t1, t2, out)
    assert out[:, :, CH_IMP].sum() > 0
    backward_dendrite(out)
    coh = coherence_map(out)
    assert float(coh.mean()) >= 0.0


def test_dendritic_engine_loop():
    eng = DendriticBrainEngine(width=64, height=64)
    frame = (_rgb_spot(64) * 255).astype(np.uint8)
    for _ in range(10):
        eng.step(frame)
    img = eng.render()
    assert img.shape == (64, 64, 3)
    assert eng.stats.tick == 10
    assert eng.stats.mean_coherence > -1.0
