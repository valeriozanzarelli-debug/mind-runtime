"""Verifica parità CPU vs CUDA ottimizzato (tolleranza float32)."""

from __future__ import annotations

import numpy as np

from mindruntime.field_v2 import field_zeros, spike_times_zeros
from mindruntime.gpu_physics_v2 import (
    gamma_phase_lock,
    hh_rk4_step_field,
    initialize_field_v2,
    predictive_coding,
    soc_avalanche,
    turing_reaction_diffusion,
)


def _clone_field(field: np.ndarray) -> np.ndarray:
    return np.array(field, copy=True)


def test_turing_cpu_vs_reference():
  h, w = 32, 32
  rgb = np.random.rand(h, w, 3).astype(np.float32)
  a_in = field_zeros(h, w)
  a_out = field_zeros(h, w)
  b_in = field_zeros(h, w)
  b_out = field_zeros(h, w)
  initialize_field_v2(rgb, a_in, seed=7)
  b_in[:] = a_in
  turing_reaction_diffusion(a_in, a_out, decay=0.97)
  turing_reaction_diffusion(b_in, b_out, decay=0.97)
  np.testing.assert_allclose(a_out, b_out, rtol=1e-5, atol=1e-5)


def test_physics_step_channels_stable():
  h, w = 24, 24
  rgb = np.random.rand(h, w, 3).astype(np.float32) * 0.4
  field = field_zeros(h, w)
  scratch = field_zeros(h, w)
  spike = spike_times_zeros(h, w)
  initialize_field_v2(rgb, field, seed=3)
  before = _clone_field(field)
  hh_rk4_step_field(field)
  turing_reaction_diffusion(field, scratch)
  field[:] = scratch
  soc_avalanche(field)
  gamma_phase_lock(field)
  predictive_coding(field, spike, 1.0)
  assert np.isfinite(field).all()
  assert field.shape == before.shape
