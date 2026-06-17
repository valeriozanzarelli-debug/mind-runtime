"""Test budget neurale e piano capacità."""

from organism.cognition.brain_budget import (
    analyze_brain,
    build_capacity_plan,
    recommend_gpu_resolution,
    recommend_graph_tier,
)
from organism.brain.topology import NeuralTopology


def test_analyze_brain_layers():
    brain = NeuralTopology(seed=1)
    brain.add_neuron("associative", "pattern_matcher")
    brain.add_neuron("motor", "motion_gesture_emitter")
    brain.add_neuron("motor", "speech_phoneme_generator")
    b = analyze_brain(brain)
    assert b.total == 3
    assert b.motor_body == 1
    assert b.motor_speech == 1


def test_gpu_resolution_8gb():
    w, h, d, mb = recommend_gpu_resolution(8192)
    assert w >= 128
    assert h >= 128
    assert d >= 8
    assert mb < 2500
    assert w <= 512


def test_gpu_resolution_3d_voxels():
    w, h, d, mb = recommend_gpu_resolution(8192, depth=128)
    voxels = w * h * d
    assert voxels >= 10_000_000


def test_recommend_tier_16gb():
    tier = recommend_graph_tier(16)
    assert tier["recommended_variant"] in ("giga", "mind_giga", "mind_compact", "mega", "ultra", "ultra_compact")


def test_capacity_plan_totals():
    plan = build_capacity_plan(
        graph_neurons=4_000_000,
        graph_thinking=2_200_000,
        graph_synapses=16_000_000,
        gpu_w=512,
        gpu_h=384,
        gpu_d=128,
    )
    assert plan.total_effective_neurons == 4_000_000 + 512 * 384 * 128
    assert plan.gpu_pixels == 512 * 384 * 128
