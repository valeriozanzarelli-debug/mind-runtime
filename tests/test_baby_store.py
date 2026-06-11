"""Persistenza baby — non rinasce ad ogni apertura."""

import json
from pathlib import Path

from organism.autonomous.baby_agent import BabyAgent
from organism.drives.curiosity import stimulus_key_from_sensory


def test_birth_idempotent_when_alive(tmp_path: Path):
    store = tmp_path / "baby.json"
    b = BabyAgent(seed=7, store_path=str(store))
    first = b.birth()
    assert first["resumed"] is False
    b.teach_repetition("ciao mondo", stimulus_key=stimulus_key_from_sensory(text="ciao"))
    again = b.birth()
    assert again["resumed"] is True
    assert b.teacher.partial(stimulus_key_from_sensory(text="ciao")) == "ciao mondo"


def test_survives_reload(tmp_path: Path):
    store = tmp_path / "baby.json"
    key = stimulus_key_from_sensory(text="rosso")

    b1 = BabyAgent(seed=8, store_path=str(store))
    b1.birth()
    for _ in range(3):
        b1.teach_repetition("questo è rosso", stimulus_key=key)
    b1.sense(text="altro")
    assert store.exists()

    b2 = BabyAgent(seed=99, store_path=str(store))
    assert b2._born is True
    assert b2.teacher.respond(key) == "questo è rosso"
    assert b2.speech.phonemes.count > 0
    assert b2.seed == 8
    payload = json.loads(store.read_text())
    assert payload["born"] is True
