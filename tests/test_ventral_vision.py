"""Flusso ventrale — V1 bordi, V2 forma, V4 colore, IT gist."""

from organism.cognition.visual_bind import VisualBinder
from organism.motor.compose_speech import SpeechComposer
from organism.cognition.thought import Thought
from organism.motor.emergent_speech import EmergentSpeechMotor
from organism.runtime import OrganismRuntime
from organism.sensory.ventral_vision import gist_similarity, process_ventral_stream


def _red_circle(size: int = 32) -> tuple[list[list[int]], list[list[tuple[int, int, int]]]]:
    gray = [[20] * size for _ in range(size)]
    rgb = [[(20, 20, 20)] * size for _ in range(size)]
    cx = cy = size // 2
    r = size // 4
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                gray[y][x] = 200
                rgb[y][x] = (220, 40, 40)
    return gray, rgb


def test_v1_v2_blob_shape():
    gray, rgb = _red_circle()
    out = process_ventral_stream(gray, rgb_grid=rgb)
    assert out["blob_count"] >= 1
    assert any("rotondo" in s for s in out["symbols"])
    assert out["features"].get("gist")


def test_v4_color_from_blob():
    gray, rgb = _red_circle()
    out = process_ventral_stream(gray, rgb_grid=rgb)
    assert out["color"] in ("rosso", "rosa", "arancione", "marrone")
    assert any(s.startswith("COL:") for s in out["symbols"])


def test_gist_similarity_same_shape():
    gray1, rgb1 = _red_circle(32)
    gray2, rgb2 = _red_circle(34)
    f1 = process_ventral_stream(gray1, rgb_grid=rgb1)["features"]
    f2 = process_ventral_stream(gray2, rgb_grid=rgb2)["features"]
    assert gist_similarity(f1, f2) >= 0.45


def test_spatial_patches_in_gist():
    gray, rgb = _red_circle(48)
    feat = process_ventral_stream(gray, rgb_grid=rgb)["features"]
    patches = feat["gist"].get("patches_lum", [])
    assert len(patches) == 16
    assert max(patches) > min(patches)


def test_patches_discriminate_layout():
    size = 48
    left = [[20] * size for _ in range(size)]
    right = [[20] * size for _ in range(size)]
    lr = [[(20, 20, 20)] * size for _ in range(size)]
    rr = [[(20, 20, 20)] * size for _ in range(size)]
    for y in range(size):
        for x in range(size // 2):
            left[y][x] = 210
            lr[y][x] = (210, 50, 50)
        for x in range(size // 2, size):
            right[y][x] = 210
            rr[y][x] = (50, 50, 210)
    f_left = process_ventral_stream(left, rgb_grid=lr)["features"]
    f_right = process_ventral_stream(right, rgb_grid=rr)["features"]
    assert gist_similarity(f_left, f_right) < gist_similarity(f_left, f_left)


def test_visual_bind_teach_and_recognize():
    vb = VisualBinder()
    gray, rgb = _red_circle()
    feat = process_ventral_stream(gray, rgb_grid=rgb)["features"]
    for _ in range(3):
        vb.bind("s1", "vedo una palla rossa", features=feat, object_sig=feat["object_sig"], object_name="palla")
    gray2, rgb2 = _red_circle(30)
    feat2 = process_ventral_stream(gray2, rgb_grid=rgb2)["features"]
    assert vb.recognize_object(feat2, min_sim=0.4) == "palla"


def test_motor_color_answer_no_template():
    comp = SpeechComposer(seed=3)
    org = OrganismRuntime.baby(seed=3)
    motor = EmergentSpeechMotor(org.brain, seed=3)
    comp.bind(org.brain, motor)
    for w in ("è", "rosso", "colore", "vedo", "palla"):
        comp.absorb(w, boost=2.5)
    t = Thought(
        themes=["è", "rosso", "colore"],
        symbols=["COL:rosso", "QUESTION:color"],
        pressure=0.6,
    )
    out = comp._from_thought_emergent(t, motor=motor)
    assert "rosso" in out.lower()
    assert out.endswith(".")
