"""Crescita naturale — parlare libero, compiti, lettura."""

from organism.autonomous.baby_agent import BabyAgent
from organism.cognition.tasks import TaskRunner
from organism.sensory.reading import ReadingChannel


def test_baby_can_speak_without_conscious_gate(tmp_path):
    b = BabyAgent(seed=5, store_path=str(tmp_path / "b.json"))
    b.birth()
    for _ in range(3):
        b.teach_dialogue("ciao", "ciao amico")
    m = b.sense(text="ciao")["moment"]
    assert m["presence"]["speaks"] is True
    assert m["spoke"] or m["wanted_to_speak"] or m["understood"]


def test_task_repeat_done(tmp_path):
    b = BabyAgent(seed=6, store_path=str(tmp_path / "t.json"))
    b.birth()
    b.assign_task("repeat", "ripeti", "ciao mondo")
    for _ in range(3):
        b.teach_dialogue("ripeti", "ciao mondo")
    m = b.sense(text="ripeti")["moment"]
    assert m.get("task", {}).get("done") or "mondo" in m.get("spoke", "")


def test_reading_channel_spikes():
    from organism.runtime import OrganismRuntime

    org = OrganismRuntime.baby(seed=7)
    ch = ReadingChannel(org.brain)
    r = ch.perceive("ciao mondo bello")
    assert r.chars > 0
    assert len(r.words) >= 2


def test_task_runner_evaluate():
    tr = TaskRunner()
    tr.assign("repeat", "dì", "buongiorno")
    fb = tr.evaluate_attempt("buongiorno a tutti")
    assert fb["active"]
    assert fb.get("done") or fb.get("score", 0) > 0
