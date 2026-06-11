"""Nursery — osservabilità, curriculum, auto-sviluppo."""

from organism.nursery import Nursery, phases_dict
from organism.nursery.server import NurseryServer


def test_birth_generates_brain():
    n = Nursery(seed=1)
    result = n.birth()
    assert result["stats"]["neurons"] > 500
    assert result["stats"]["synapses"] > 10_000
    assert n.journal.birth_record is not None
    assert len(result["graph"]["nodes"]) >= 0


def test_teach_records_thoughts():
    n = Nursery(seed=2)
    n.birth()
    r = n.teach({"text": "lampadina non si accende"}, phase="language")
    assert len(n.journal.entries) == 1
    assert r["entry"]["action"] == "replace_bulb"
    assert len(r["entry"]["thoughts"]) > 0


def test_curriculum_verifies_auto_development():
    n = Nursery(seed=3)
    n.run_full_curriculum()
    v = n.verify()
    assert v["auto_development"] is True
    assert len(n.growth.timeline) >= 5
    checks_ok = sum(1 for c in v["checks"] if c["ok"])
    assert checks_ok >= 4


def test_phases_dict():
    phases = phases_dict()
    assert any(p["id"] == "vision" for p in phases)
    assert any(p["id"] == "language" for p in phases)


def test_thought_stream_readable():
    n = Nursery(seed=4)
    n.birth()
    n.teach({"text": "ciao"}, phase="language")
    lines = n.journal.stream_text()
    assert any("stimolo" in l for l in lines)


def test_graph_export_after_stimulus():
    n = Nursery(seed=5)
    n.birth()
    n.teach({"shapes": "quadrato+cerchio,triangolo+cerchio,rettangolo+"}, phase="vision")
    g = n.org.brain.export_active_subgraph()
    assert g["meta"]["total_neurons"] > 0


def test_nursery_server_state():
    srv = NurseryServer(port=18765, seed=6)
    srv.nursery.birth()
    state = srv.nursery.state()
    assert state["born"] is True
    assert "thoughts" in state
    assert "growth" in state
