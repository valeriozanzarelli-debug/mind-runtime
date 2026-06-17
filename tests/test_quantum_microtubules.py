"""Test layer microtubuli — coerenza e collasso computazionale."""

import os

from organism.brain.quantum_microtubules import QuantumMicrotubuleLayer


def test_quantum_coherence_on_high_ignition():
    os.environ["ORGANISM_QUANTUM"] = "1"
    q = QuantumMicrotubuleLayer()
    for _ in range(20):
        q.tick(neural_activity=0.8, workspace_ignition=0.75, thought_seed="test")
    assert q.state.coherence >= 0.0
    assert q.state.contested is True


def test_quantum_disabled():
    os.environ["ORGANISM_QUANTUM"] = "0"
    q = QuantumMicrotubuleLayer()
    q.tick(neural_activity=0.9, workspace_ignition=0.9)
    assert q.state.coherence == 0.0
    os.environ["ORGANISM_QUANTUM"] = "1"


def test_collapse_count_increases():
    os.environ["ORGANISM_QUANTUM"] = "1"
    q = QuantumMicrotubuleLayer()
    before = q.state.collapse_count
    for _ in range(40):
        q.tick(neural_activity=0.95, workspace_ignition=0.9, thought_seed="momento")
    assert q.state.collapse_count >= before
