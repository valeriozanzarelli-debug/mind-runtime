"""Colori, sì/no, risposte emergenti."""

from organism.motor.compose_speech import SpeechComposer
from organism.cognition.thought import Thought
from organism.sensory.color_scene import analyze_color_rgb, rgb_to_color_name


def test_rgb_red():
    assert rgb_to_color_name(200, 30, 30) == "rosso"


def test_rgb_green():
    assert rgb_to_color_name(30, 200, 40) == "verde"


def test_analyze_color():
    col = analyze_color_rgb(200, 30, 30)
    assert col["color"] == "rosso"
    assert "COL:rosso" in col["symbols"]


def test_emergent_yes_no():
    c = SpeechComposer(seed=1)
    for w in ("sì", "no", "vedo", "rosso", "cerchio"):
        c.absorb(w, boost=2.0)
    t = Thought(themes=["no"], pressure=0.5)
    out = c._from_thought_emergent(t)
    assert out
    assert "no" in out.lower()


def test_emergent_color_answer():
    from organism.motor.emergent_speech import EmergentSpeechMotor
    from organism.runtime import OrganismRuntime

    c = SpeechComposer(seed=2)
    org = OrganismRuntime.baby(seed=2)
    motor = EmergentSpeechMotor(org.brain, seed=2)
    c.bind(org.brain, motor)
    for w in ("è", "rosso", "verde", "vedo", "colore"):
        c.absorb(w, boost=2.0)
    t = Thought(
        themes=["è", "rosso", "colore"],
        symbols=["COL:rosso", "QUESTION:color"],
        pressure=0.6,
    )
    out = c._from_thought_emergent(t, motor=motor)
    assert "rosso" in out.lower()


def test_visual_bind_color_theme():
    from organism.cognition.visual_bind import VisualBinder

    vb = VisualBinder()
    themes = vb.themes("x", {"luminance": 0.5, "contrast": 0.2, "color": "rosso"})
    assert "rosso" in themes
    assert "COL:rosso" in themes
