import time

from cerebrum.brain import BrainConfig, Cerebrum
from cerebrum.body import Homeostasis, Neurochemistry, Drives, NeonatalReflexes
from cerebrum.neuro import NeuralField
from cerebrum.neuro.field import FieldConfig
from cerebrum.sense import VisionSense, LanguageSense
from cerebrum.motor import SpeechMotor


def test_field_steps_and_phi():
    f = NeuralField(FieldConfig(neurons=256))
    chem = Neurochemistry().as_dict()
    for _ in range(50):
        m = f.step(None, chem)
    assert m["rate"] >= 0.0
    assert f.phi_estimate() >= 0.0
    assert f.computational_units == 256 * 5


def test_homeostasis_distress_and_care():
    h = Homeostasis()
    h.satiety = 0.0
    assert h.distress() > 0.3
    h.feed(0.5)
    assert h.satiety > 0.0


def test_reflexes_fire_on_startle():
    r = NeonatalReflexes()
    fired = r.evaluate({"intensity": 0.9, "motion": 0.5, "brightness": 0.7},
                       {"distress": 0.7})
    names = {f["reflex"] for f in fired}
    assert "moro" in names
    assert "crying" in names


def test_language_builds_vocabulary():
    l = LanguageSense()
    enc = l.encode("ciao mondo ciao")
    assert enc["tokens"]
    assert l.vocabulary_size() == 2


def test_vision_detects_brightness_and_motion():
    v = VisionSense()
    out = v.process(stats={"brightness": 0.8, "motion": 0.4})
    assert out["active"] is True
    assert out["brightness"] == 0.8


def test_speech_babbles():
    s = SpeechMotor()
    u = s.utter(emotion="quiete", drive="curiosità", activity=0.1, known_tokens=[], urge=0.5)
    assert isinstance(u, str) and len(u) > 0


def test_brain_lifecycle_and_chat():
    b = Cerebrum(BrainConfig(neurons=256))
    b.birth()
    try:
        time.sleep(0.5)
        assert b.alive
        res = b.respond("ciao piccolo")
        assert "reply" in res
        h = b.health()
        assert h["neurons"] == 256
        intro = b.introspect()
        assert "neurochemistry" in intro
        assert intro["field"]["neurons"] == 256
    finally:
        b.shutdown()
    assert not b.alive


def test_brain_reacts_to_webcam():
    b = Cerebrum(BrainConfig(neurons=256))
    b.birth()
    try:
        b.see(stats={"brightness": 0.9, "motion": 0.6})
        time.sleep(0.3)
        assert b.webcam_active is True
    finally:
        b.shutdown()
