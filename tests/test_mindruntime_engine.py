"""Test mindruntime — CPU fallback (CI senza GPU)."""

import numpy as np

from mindruntime.ai_trainer import AITrainer
from mindruntime.cuda_util import cuda_info
from mindruntime.gpu_core import (
    initialize_neurons,
    match_resonators,
    propagate_wavefront,
    update_weights_hebbian,
)
from mindruntime.gpu_engine import GPUBrainEngine
from mindruntime.gpu_physics import (
    CH_COH,
    CH_IMP,
    N_CHANNELS,
    bifurcation_phase_lock,
    hodgkin_huxley_step,
    initialize_field,
    initialize_hh_state,
    phase_gradient_memory,
    soc_criticality_tunning,
    turing_reaction_diffusion,
)
from mindruntime.resonators import (
    TEMPLATE_NAMES,
    build_resonator_bank,
    correlate_resonators,
    load_resonators_from_disk,
    spatial_template,
)


def _rgb_spot(size: int = 64) -> np.ndarray:
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    c = size // 2
    for y in range(size):
        for x in range(size):
            if (x - c) ** 2 + (y - c) ** 2 < (size // 5) ** 2:
                rgb[y, x] = 0.9
    return rgb


def test_cuda_info_returns_dict():
    info = cuda_info()
    assert "numba" in info
    assert "cuda" in info


def test_initialize_neurons():
    h, w = 64, 64
    neurons = np.zeros((h, w, 4), dtype=np.float32)
    initialize_neurons(_rgb_spot(h), neurons, seed=1)
    assert neurons[:, :, 0].max() > 0.1
    assert neurons[:, :, 2].mean() > 0.3


def test_propagate_changes_state():
    h, w = 48, 48
    s = np.zeros((h, w, 4), dtype=np.float32)
    initialize_neurons(_rgb_spot(h), s, seed=2)
    t1 = s.copy()
    t2 = s.copy()
    out = np.zeros_like(s)
    for _ in range(5):
        propagate_wavefront(t1, t2, out)
        t2, t1 = t1, out.copy()
    assert out[:, :, 0].sum() > 0


def test_hebbian_increases_weights():
    h, w = 32, 32
    state = np.zeros((h, w, 4), dtype=np.float32)
    initialize_neurons(_rgb_spot(h), state, seed=3)
    w0 = state[:, :, 2].mean()
    for _ in range(10):
        update_weights_hebbian(state)
    assert state[:, :, 2].mean() >= w0


def test_resonator_bank():
    bank = build_resonator_bank(TEMPLATE_NAMES[:6], size=24)
    assert bank["stack"].shape[0] == 6
    circle = spatial_template("circle", 24)
    assert circle.max() > 0.5


def test_match_resonators():
    bank = build_resonator_bank(("circle", "square"), size=32)
    imp = bank["spatial"]["circle"]
    pad = np.zeros((64, 64), dtype=np.float32)
    pad[16:48, 16:48] = imp
    scores = match_resonators(pad, bank["stack"])
    assert scores[0] > scores[1]


def test_physics_initialize_field():
    h, w = 48, 48
    field = np.zeros((h, w, N_CHANNELS), dtype=np.float32)
    initialize_field(_rgb_spot(h), field, threshold=0.3, seed=1)
    assert field[:, :, CH_IMP].max() > 0.1
    assert field[:, :, CH_COH].sum() == 0.0


def test_physics_pipeline_step():
    h, w = 32, 32
    field = np.zeros((h, w, N_CHANNELS), dtype=np.float32)
    hh = np.zeros((h, w, 4), dtype=np.float32)
    initialize_field(_rgb_spot(h), field, seed=2)
    initialize_hh_state(hh)
    avalanche = np.zeros((h, w), dtype=np.float32)
    persistence = np.zeros((h, w), dtype=np.float32)
    prev1 = field.copy()
    prev2 = field.copy()
    for _ in range(5):
        hodgkin_huxley_step(field, hh)
        out = np.zeros_like(field)
        turing_reaction_diffusion(field, prev1, prev2, hh, out)
        prev2, prev1, field = prev1, field, out
        bifurcation_phase_lock(field)
    soc_criticality_tunning(field, avalanche)
    phase_gradient_memory(field, persistence)
    assert field[:, :, CH_IMP].sum() > 0


def test_load_resonators_from_disk():
    res = load_resonators_from_disk(size=16)
    assert len(res) == 28  # circle + square + 26 letters
    symbols = {r["symbol"] for r in res}
    assert "●" in symbols
    assert "A" in symbols
    assert "Z" in symbols


def test_correlate_resonators():
    res = load_resonators_from_disk(size=24)
    circle = next(r for r in res if r["name"] == "circle")
    pad = np.zeros((64, 64), dtype=np.float32)
    pad[20:44, 20:44] = circle["pattern"]
    scores = correlate_resonators(pad, res[:3])
    assert scores[0][1] > scores[1][1]


def test_engine_step_and_render():
    engine = GPUBrainEngine(width=64, height=64, memory_every=5, soc_every=5)
    frame = (_rgb_spot(64) * 255).astype(np.uint8)
    for _ in range(12):
        engine.step(frame)
    img = engine.render()
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    state = engine.export_state_for_training()
    assert "impulse" in state
    assert state["impulse"].shape == (64, 64)
    assert "coherence" in state


def test_engine_export_import_server():
    engine = GPUBrainEngine(width=32, height=32)
    engine.step((_rgb_spot(32) * 255).astype(np.uint8))
    for _ in range(3):
        engine.step()
    blob = engine.export_state_for_server()
    engine2 = GPUBrainEngine(width=32, height=32)
    engine2.import_state_from_server(blob)
    assert engine2.step_count == engine.step_count
    assert engine2.field_t.shape == engine.field_t.shape


def test_ai_trainer_stub_without_torch_label():
    engine = GPUBrainEngine(width=32, height=32)
    engine.step((_rgb_spot(32) * 255).astype(np.uint8))
    trainer = AITrainer()
    result = trainer.train_step(engine.export_state_for_training(), label=None)
    assert result.weight_delta is None
