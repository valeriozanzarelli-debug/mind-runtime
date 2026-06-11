from organism.sensory.vision_sense import VisionSense


def _grid(fill: int = 128, size: int = 32) -> list[list[int]]:
    return [[fill] * size for _ in range(size)]


def test_motion_skip_similar_frames():
    vs = VisionSense()
    g1 = _grid(100)
    g2 = _grid(102)
    vs.process(g1)
    assert vs.should_skip_frame(g2)


def test_hierarchical_features_on_contrast():
    vs = VisionSense()
    g = _grid(30)
    for y in range(8, 24):
        for x in range(8, 24):
            g[y][x] = 220
    r = vs.process(g, force=True)
    feat = r["features"]
    assert feat.get("edge_density", 0) > 0
    assert feat.get("shapes")
