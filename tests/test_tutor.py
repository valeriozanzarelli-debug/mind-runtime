"""Test agente tutore."""

from organism.tutor.tutor_agent import TutorAgent, tutor_state_path


class _FakeBaby:
    def __init__(self) -> None:
        self._born = False
        self._cycles = 0
        self.foundation_calls = 0
        self.cycle_calls = 0

    def birth(self) -> dict:
        self._born = True
        return {"ok": True, "stats": {"neurons": 1500}}

    def train_foundation(self, *, repeats: int = 1) -> dict:
        self.foundation_calls += 1
        return {"ok": True, "cycles": repeats}

    def train_integrated_cycle(self, cycle: int) -> dict:
        self.cycle_calls += 1
        return {"cycle": cycle, "probe": "ciao", "spoke_words": 2}

    def sleep_cycle(self) -> dict:
        return {"pruned": 1}


def test_tutor_birth_then_foundation(tmp_path, monkeypatch):
    state_file = tmp_path / "tutor.json"
    monkeypatch.setattr("organism.tutor.tutor_agent.tutor_state_path", lambda: state_file)
    baby = _FakeBaby()
    tutor = TutorAgent(baby_fn=lambda: baby, state_path=state_file)

    r1 = tutor.tick_once()
    assert r1["step"]["action"] == "birth"
    assert baby._born

    r2 = tutor.tick_once()
    assert r2["step"]["action"] == "foundation"
    assert baby.foundation_calls == 1
    assert tutor.state.foundation_done

    r3 = tutor.tick_once()
    assert r3["step"]["action"] == "cycle"
    assert baby.cycle_calls == 1
    assert tutor.state.cycle == 1


def test_tutor_start_stop():
    baby = _FakeBaby()
    baby._born = True
    tutor = TutorAgent(baby_fn=lambda: baby)
    tutor.state.foundation_done = True

    status = tutor.start(interval_s=0.1)
    assert status["tutor"]["status"] == "running"
    import time
    time.sleep(0.35)
    tutor.stop()
    assert tutor.state.cycle >= 1


def test_tutor_persists_state(tmp_path, monkeypatch):
    state_file = tmp_path / "tutor.json"
    monkeypatch.setattr("organism.tutor.tutor_agent.tutor_state_path", lambda: state_file)
    baby = _FakeBaby()
    baby._born = True
    t1 = TutorAgent(baby_fn=lambda: baby, state_path=state_file)
    t1.state.foundation_done = True
    t1.state.cycle = 42
    t1._persist_state()

    t2 = TutorAgent(baby_fn=lambda: baby, state_path=state_file)
    assert t2.state.cycle == 42
    assert t2.state.foundation_done is True
