"""Riconoscimento oggetti — forme e binding."""

from organism.cognition.visual_bind import VisualBinder
from organism.sensory.object_scene import analyze_objects, feature_similarity


def _circle_grid(size: int = 32) -> list[list[int]]:
    grid = [[20] * size for _ in range(size)]
    cx = cy = size // 2
    r = size // 4
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                grid[y][x] = 220
    return grid


def _square_grid(size: int = 32) -> list[list[int]]:
    grid = [[15] * size for _ in range(size)]
    m = size // 4
    for y in range(m, size - m):
        for x in range(m, size - m):
            grid[y][x] = 230
    return grid


def test_analyze_circle():
    obj = analyze_objects(_circle_grid())
    assert obj["blob_count"] >= 1
    assert any("rotondo" in s for s in obj["symbols"])


def test_analyze_square():
    obj = analyze_objects(_square_grid())
    assert obj["blob_count"] >= 1


def test_visual_binder_object_recognition():
    vb = VisualBinder()
    g1 = _circle_grid()
    o1 = analyze_objects(g1)
    for _ in range(3):
        vb.bind("sig1", "vedo un cerchio", features=o1["features"], object_sig=o1["object_sig"], object_name="cerchio")
    g2 = _circle_grid(34)
    o2 = analyze_objects(g2)
    name = vb.recognize_object(o2["features"])
    assert name == "cerchio"


def test_feature_similarity():
    a = {"luminance": 0.8, "edge_mass": 0.2, "blob_count": 1, "shapes": ["rotondo"]}
    b = {"luminance": 0.75, "edge_mass": 0.22, "blob_count": 1, "shapes": ["rotondo"]}
    assert feature_similarity(a, b) >= 0.5
