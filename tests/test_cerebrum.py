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


def test_field_has_regions_and_sparse_synapses():
    f = NeuralField(FieldConfig(neurons=512))
    # regioni distinte con ruoli
    assert set(f.regions) == {"sensory", "thalamus", "association", "hippocampus", "prefrontal"}
    # sinapsi sparse (molte meno di N^2)
    assert 0 < f.synapses < 512 * 512
    named = f.region_activity_named()
    assert "association" in named


def test_field_three_factor_plasticity_changes_weights():
    f = NeuralField(FieldConfig(neurons=256))
    import numpy as np
    w0 = f._w_np.copy() if not f._use_torch else None
    chem = Neurochemistry().as_dict()
    chem["dopamine"] = 0.9  # forte terzo fattore
    ext = np.ones(64, dtype=np.float32)
    for _ in range(80):
        f.step(ext, chem)
    if w0 is not None:
        assert not np.allclose(w0, f.w)


def test_predictive_coder_learns():
    from cerebrum.mind import PredictiveCoder
    import numpy as np
    pc = PredictiveCoder(assoc_dim=16, sensory_dim=16)
    assoc = np.ones(16, dtype=np.float32) * 0.5
    target = np.ones(16, dtype=np.float32) * 0.8
    first = pc.learn(assoc, target)
    for _ in range(200):
        pc.learn(assoc, target)
    last = pc.learn(assoc, target)
    assert last < first  # la sorpresa cala imparando


def test_development_advances_with_experience():
    from cerebrum.mind import Development
    d = Development()
    assert d.level == 0
    d.update(age_ticks=500, vocabulary=1, episodes=0, mean_activity=0.1)
    assert d.level >= 1
    d.update(age_ticks=4000, vocabulary=20, episodes=30, mean_activity=0.1)
    assert d.level >= 2


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


def test_speech_stage_progression():
    s = SpeechMotor()
    # stadio 0 neonato: solo versi, mai parole conosciute
    for _ in range(20):
        u = s.utter(emotion="quiete", drive="curiosità", activity=0.1,
                    known_tokens=["mamma", "pappa"], urge=0.5, stage=0)
        assert u in {"uè", "aa", "eh", "ah", "ngh", "uè uè"} or "!" in u
    # stadio 4: puo' combinare due parole conosciute
    combos = 0
    for _ in range(60):
        u = s.utter(emotion="quiete", drive="curiosità", activity=0.2,
                    known_tokens=["mamma", "pappa", "ciao"], urge=0.9, stage=4)
        if len(u.split()) == 2:
            combos += 1
    assert combos > 0


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
        assert "stage" in h
        intro = b.introspect()
        assert "neurochemistry" in intro
        assert intro["field"]["neurons"] == 256
        assert "development" in intro
        assert "prediction" in intro
        assert "regions" in intro["field"]
    finally:
        b.shutdown()
    assert not b.alive


def test_brain_thinks_without_stimuli():
    # senza stimoli il cervello deve comunque generare pensieri (immaginazione)
    b = Cerebrum(BrainConfig(neurons=256))
    b.birth()
    try:
        time.sleep(1.0)
        moments = b.consciousness(n=50)["moments"]
        assert len(moments) > 0
    finally:
        b.shutdown()


def test_brain_reacts_to_webcam():
    b = Cerebrum(BrainConfig(neurons=256))
    b.birth()
    try:
        b.see(stats={"brightness": 0.9, "motion": 0.6})
        time.sleep(0.3)
        assert b.webcam_active is True
    finally:
        b.shutdown()
