"""Dialogo — associazione quando→risposta."""

from organism.autonomous.baby_agent import BabyAgent
from organism.teaching.dialogue import DialogueTeacher


def test_dialogue_pair_not_echo_wrong(tmp_path):
    d = DialogueTeacher(consolidate_at=1)
    d.teach("ciao", "ciao")
    d.teach("come stai", "sto bene")
    r1, _ = d.respond("ciao")
    r2, _ = d.respond("come stai")
    assert r1 == "ciao"
    assert r2 == "sto bene"
    assert r2 != "ciao, come stai"


def test_agent_dialogue_through_sense(tmp_path):
    store = str(tmp_path / "b.json")
    b = BabyAgent(seed=1, store_path=store)
    b.birth()
    for _ in range(3):
        b.teach_dialogue("come stai", "sto bene")
    m = b.sense(text="come stai")["moment"]
    assert m["understood"] or m.get("from_thought")
    spoke = (m.get("spoke") or "").lower()
    assert "sto" in spoke or "bene" in spoke or "come" in spoke


def test_agent_code_pair(tmp_path):
    store = str(tmp_path / "c.json")
    b = BabyAgent(seed=2, store_path=store)
    b.birth()
    for _ in range(3):
        b.teach_dialogue("stampa ciao", 'print("ciao")', kind="code")
    m = b.sense(text="stampa ciao")["moment"]
    assert m["understood"]
    assert "print" in m["code"]
