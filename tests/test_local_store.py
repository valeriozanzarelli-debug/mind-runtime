"""Test persistenza locale mindruntime."""

import json
from pathlib import Path

import numpy as np

from mindruntime.dendritic_engine import DendriticBrainEngine
from mindruntime.local_store import append_tick, load_last_state, save_snapshot, store_dir


def test_store_dir_created(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANISM_DATA_DIR", str(tmp_path))
    d = store_dir()
    assert d.is_dir()
    assert "mindruntime" in str(d)


def test_save_and_load_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANISM_DATA_DIR", str(tmp_path))
    eng = DendriticBrainEngine(width=32, height=32)
    spot = np.zeros((32, 32, 3), dtype=np.uint8)
    spot[14:18, 14:18] = 200
    eng.step(spot)
    p = save_snapshot(eng, frames=1, fps=10.0)
    assert p.is_file()
    data = load_last_state()
    assert data is not None
    assert data["tick"] == 1


def test_journal_append(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANISM_DATA_DIR", str(tmp_path))
    append_tick({"conscious": True, "order": 0.6})
    lines = (store_dir() / "session.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["conscious"] is True
