"""Mare e distese — nomi appresi, non rettangolo blu."""

from organism.cognition.visual_bind import VisualBinder
from organism.sensory.ventral_vision import gist_similarity, process_ventral_stream
from organism.teaching.vision_phrase import parse_vision_teaching


def _horizon_sea(size: int = 96) -> tuple[list[list[int]], list[list[tuple[int, int, int]]]]:
    gray = [[20] * size for _ in range(size)]
    rgb = [[(20, 20, 20)] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            if y < size // 3:
                gray[y][x] = 200
                rgb[y][x] = (180, 200, 230)
            else:
                gray[y][x] = 70 + (x % 20)
                rgb[y][x] = (30, 110, 190)
    return gray, rgb


def test_horizon_not_quadrato():
    gray, rgb = _horizon_sea()
    out = process_ventral_stream(gray, rgb_grid=rgb)
    feat = out["features"]
    assert feat.get("scene_type") in ("orizzonte", "distesa")
    assert "quadrato" not in feat.get("shapes", [])


def test_mare_teaching_recognized():
    vb = VisualBinder()
    gray, rgb = _horizon_sea(64)
    feat = process_ventral_stream(gray, rgb_grid=rgb)["features"]
    for _ in range(4):
        vb.bind(
            "sig1",
            "questo è il mare",
            features=feat,
            object_sig=feat["object_sig"],
            object_name="mare",
        )
    gray2, rgb2 = _horizon_sea(66)
    feat2 = process_ventral_stream(gray2, rgb_grid=rgb2)["features"]
    assert vb.recognize_object(feat2, min_sim=0.36) == "mare"
    assert vb.percept_words(feat2) == ["mare"]
    assert "blu" not in vb.speech_themes(feat2)
    assert "quadrato" not in vb.speech_themes(feat2)


def test_parse_il_mare():
    p = parse_vision_teaching("questo è il mare")
    assert p.object_name == "mare"
