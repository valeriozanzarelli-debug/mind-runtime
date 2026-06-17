"""Test scaling 3D + neuroni compatti."""

from __future__ import annotations

import os

import pytest

from organism.brain.compact_store import CompactNeuralBackend, estimate_bytes_per_neuron
from organism.brain.impulse_field_3d import ImpulseField3D, create_impulse_field_3d
from organism.brain.impulse_field import create_impulse_field


def test_impulse_field_3d_neuron_count():
    f = ImpulseField3D(width=64, height=48, depth=32, device="numpy")
    assert f.neuron_count == 64 * 48 * 32
    assert f.dimensions == "64x48x32"
    f.inject_text_energy([0.5, 0.3, 0.8])
    f.step(steps=2)
    sig = f.signature()
    assert len(sig) == 128
    st = f.stats()
    assert st["spatial"] == "3d"
    assert st["voxels"] == f.neuron_count


def test_create_impulse_field_uses_depth_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORGANISM_IMPULSE_D", "32")
    monkeypatch.setenv("ORGANISM_TEMPORAL", "0")
    f = create_impulse_field(64, 48, device="numpy", depth=32)
    assert f.neuron_count == 64 * 48 * 32


def test_compact_backend_bulk():
    c = CompactNeuralBackend()
    ids = c.add_neurons(1000, "associative", "pattern_matcher", 5000)
    assert len(ids) == 5000
    assert c.count == 5000
    c.set_activation(ids[0], 0.9)
    assert c.get_activation(ids[0]) == pytest.approx(0.9)
    st = c.stats()
    assert st["compact_neurons"] == 5000
    assert estimate_bytes_per_neuron() == 48


def test_compact_ram_estimate_80m():
    """80M neuroni compact ≈ 3.8 GB — entra in 16 GB server."""
    mb = 80_000_000 * estimate_bytes_per_neuron() / (1024 * 1024)
    assert mb < 4500
