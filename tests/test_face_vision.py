"""Volti ed emozioni — analisi e binding."""

from organism.cognition.face_bind import FaceEmotionBinder
from organism.sensory.face_vision import analyze_face, face_gist_similarity


def _fake_face_grid(size: int = 64) -> tuple[list[list[int]], list[list[tuple[int, int, int]]]]:
    gray = [[30] * size for _ in range(size)]
    rgb: list[list[tuple[int, int, int]]] = [[(30, 25, 20)] * size for _ in range(size)]
    cy, cx = size // 2, size // 2
    r = size // 5
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                gray[y][x] = 200
                rgb[y][x] = (220, 170, 140)
            elif (x - cx + 8) ** 2 + (y - cy + 10) ** 2 <= 16:
                gray[y][x] = 40
                rgb[y][x] = (40, 30, 25)
            elif (x - cx - 8) ** 2 + (y - cy + 10) ** 2 <= 16:
                gray[y][x] = 40
                rgb[y][x] = (40, 30, 25)
    return gray, rgb


def test_face_detection_on_synthetic():
    gray, rgb = _fake_face_grid()
    out = analyze_face(gray, rgb_grid=rgb)
    assert out["face_score"] > 0.2
    assert "gist" in out


def test_face_emotion_binder_learns():
    gray, rgb = _fake_face_grid()
    face = analyze_face(gray, rgb_grid=rgb)
    b = FaceEmotionBinder()
    for _ in range(3):
        b.teach_emotion("felice", face)
    emo = b.recognize_emotion(face)
    assert emo == "felice" or b.infer_emotion_hint(face) is not None


def test_gist_similarity_identical():
    g = {"skin_ratio": 0.3, "face_aspect": 1.0, "eye_contrast": 0.2}
    assert face_gist_similarity(g, dict(g)) > 0.95
