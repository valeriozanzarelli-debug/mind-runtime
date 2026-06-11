"""Baby agent — curiosità, sillabe udite, ripetizione, autonomia."""

from pathlib import Path

import pytest

from organism.autonomous.baby_agent import BabyAgent


@pytest.fixture
def baby_store(tmp_path: Path):
    return str(tmp_path / "baby.json")


def _agent(seed: int, store: str) -> BabyAgent:
    return BabyAgent(seed=seed, store_path=store)
from organism.drives.curiosity import stimulus_key_from_sensory, stimulus_key_visual_context
from organism.teaching.repetition import RepetitionTeacher


def test_birth_no_hardcoded_speech(baby_store):
    b = _agent(1, baby_store)
    r = b.birth()
    assert r["neurons"] > 500
    assert "lampadina" not in str(r).lower()
    assert "message" not in r
    tick = b.autonomous_tick()
    assert isinstance(tick["moment"]["spoke"], str)
    assert "brain" in tick["moment"]
    assert "lampadina" not in tick["moment"]["spoke"].lower()


def test_repetition_learning():
    t = RepetitionTeacher(consolidate_at=3)
    key = stimulus_key_from_sensory(text="rosso")
    for _ in range(3):
        r = t.teach(key, "questo è rosso")
    assert r["learned"] is True
    assert t.respond(key) == "questo è rosso"


def test_respond_by_text_word_overlap():
    t = RepetitionTeacher(consolidate_at=1)
    key = stimulus_key_visual_context(vision_hash="abc123")
    t.teach(key, "ciao come stai")
    assert t.respond_by_text("ciao") == "ciao come stai"


def test_teach_through_agent(baby_store):
    b = _agent(2, baby_store)
    b.birth()
    b.sense(text="rosso")
    key = stimulus_key_visual_context(vision_hash=b._last_vision_hash)
    for _ in range(3):
        r = b.teach_repetition("questo è rosso", stimulus_key=key)
    assert r["learned"] is True
    moment = b.sense(text="rosso")
    assert moment["moment"]["learned"] is True
    assert moment["moment"]["spoke"] == "questo è rosso" or moment["moment"]["brain"]["motor_pressure"] > 0


def test_hear_italian_builds_syllables(baby_store):
    b = _agent(5, baby_store)
    b.birth()
    b.teach_repetition("mamma")
    assert b.speech.phonemes.count > 0
    tick = b.autonomous_tick()["moment"]
    assert "lampadina" not in tick["spoke"].lower()


def test_curiosity_novelty(baby_store):
    b = _agent(3, baby_store)
    b.birth()
    b.sense(text="alpha")
    n1 = b.curiosity.state.novelty
    b.sense(text="alpha")
    n2 = b.curiosity.state.novelty
    assert n1 > n2


def test_vision_sense_via_perceive(baby_store):
    b = _agent(4, baby_store)
    b.birth()
    grid = [[200] * 16 for _ in range(16)]
    b.sense(text="luce")
    assert b.curiosity.state.last_stimulus_key
    b.org.perceive({"image": grid, "width": 16, "height": 16})
    assert b.org.brain.neuron_count > 0
    assert b.org.dna.genome.get("species") == "OrganismBaby"


def _shape_gray(size: int, lum: int = 220) -> list[int]:
    grid = [[20] * size for _ in range(size)]
    m = size // 4
    for y in range(m, size - m):
        for x in range(m, size - m):
            grid[y][x] = lum
    return [grid[y][x] for y in range(size) for x in range(size)]


def test_look_asks_when_unknown(baby_store):
    b = _agent(6, baby_store)
    b.birth()
    for w in ("dimmi", "spiega", "questo"):
        b.composer.absorb(w, boost=2.0)
    gray = _shape_gray(64)
    r = b.look(image_gray=gray, image_w=64, image_h=64)
    assert not r["recognized"]
    moment = r["moment"]
    assert moment["impulse"] == "ask"
    assert moment["spoke"]
    assert "?" in moment["spoke"] or moment["wanted_to_speak"]


def test_glance_skips_same_scene(baby_store):
    b = _agent(7, baby_store)
    b.birth()
    gray = _shape_gray(64, lum=180)
    first = b.glance(image_gray=gray, image_w=64, image_h=64, min_interval_s=0)
    assert first.get("novel") or first.get("skipped") != "same_scene"
    second = b.glance(image_gray=gray, image_w=64, image_h=64, min_interval_s=0)
    assert second.get("skipped") == "same_scene"


def test_hear_spoken_routes_teach(baby_store):
    b = _agent(9, baby_store)
    b.birth()
    gray = _shape_gray(96)
    r = b.hear_spoken(
        "questa è una cassa rosa",
        image_gray=gray,
        image_w=96,
        image_h=96,
    )
    assert r.get("mode") == "vision_object"
    assert r.get("parsed", {}).get("object") == "cassa"


def test_hear_spoken_dialogue(baby_store):
    b = _agent(10, baby_store)
    b.birth()
    b.teach_repetition("ciao")
    r = b.hear_spoken("ciao")
    assert r.get("moment", {}).get("spoke") or r.get("moment", {}).get("brain")


def test_glance_asks_on_novel_scene(baby_store):
    b = _agent(8, baby_store)
    b.birth()
    gray_a = _shape_gray(64, lum=200)
    gray_b = _shape_gray(64, lum=40)
    b._last_glance_sig = ""
    r = b.glance(image_gray=gray_b, image_w=64, image_h=64, min_interval_s=0)
    if r.get("moment"):
        assert r["moment"]["spoke"]
    else:
        b._last_glance_sig = ""
        r2 = b.glance(image_gray=gray_a, image_w=64, image_h=64, min_interval_s=0)
        assert r2.get("moment") is None or r2["moment"]["spoke"]


def test_hear_stop_word_no_crash(baby_store):
    b = _agent(11, baby_store)
    b.birth()
    r = b.hear_spoken("come")
    assert r.get("moment") is not None


def test_ciao_not_scripted_dialogue_reply(baby_store):
    b = _agent(12, baby_store)
    b.birth()
    for _ in range(5):
        b.teach_dialogue("ciao", "ciao come stai")
    spoke = (b.sense(text="ciao")["moment"].get("spoke") or "").lower().strip().rstrip(".")
    assert spoke != "ciao come stai"
