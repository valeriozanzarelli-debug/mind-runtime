"""Test architettura v2 — vault disco, impulso ibrido, orchestratore."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from organism.cognition.disk_vault import DiskMemoryVault
from organism.distributed.brain_orchestrator import BrainOrchestrator
from organism.distributed.hybrid_impulse import HybridImpulseScaffold


@pytest.fixture
def vault_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORGANISM_DISK_VAULT", str(tmp_path / "vault"))
    return tmp_path / "vault"


def test_disk_vault_append_and_search(vault_dir: Path) -> None:
    v = DiskMemoryVault()
    v.append_episode(heard="ciao mondo", spoke="ciao, ti ascolto", themes=["saluto"])
    v.append_episode(heard="python codice", spoke="scrivo una funzione", themes=["codice"])
    hits = v.search("python funzione", limit=2)
    assert len(hits) >= 1
    assert "python" in hits[0]["heard"].lower() or "funzione" in hits[0]["spoke"].lower()
    st = v.stats()
    assert st["episodes"] == 2
    assert st["path"] == str(vault_dir)


def test_disk_vault_env_flag_not_used_as_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORGANISM_DISK_VAULT", "1")
    v = DiskMemoryVault(root=tmp_path / "custom")
    v.append_episode(spoke="test")
    assert v.stats()["episodes"] == 1


def test_hybrid_fallback_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORGANISM_GPU_REMOTE", "http://127.0.0.1:59999")
    monkeypatch.setenv("ORGANISM_IMPULSE", "1")
    hybrid = HybridImpulseScaffold(health_ttl_s=0.0)
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        reading = hybrid.pulse(steps=1)
    assert reading is not None
    st = hybrid.stats()
    assert st["hybrid_mode"] == "local_fallback"


def test_brain_orchestrator_capacity() -> None:
    brain = MagicMock()
    brain.neurons = {}
    brain.synapses = {}
    brain.neuron_count = 100
    brain.synapse_count = 400

    with patch("organism.distributed.brain_orchestrator.analyze_brain") as mock_analyze:
        mock_budget = MagicMock()
        mock_budget.total = 100
        mock_budget.thinking = 60
        mock_budget.synapses = 400
        mock_budget.to_dict.return_value = {"total": 100, "thinking": 60}
        mock_analyze.return_value = mock_budget

        orch = BrainOrchestrator(brain=brain, impulse=None, disk_vault=None)
        cap = orch.capacity()
        assert cap["architecture"] == "hybrid_v2"
        assert cap["total_compute_units"] > 0
        score = orch.architecture_score()
        assert "overall" in score
        assert "gaps" in score


def test_gpu_worker_batch_and_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORGANISM_GPU_MEMORY", str(tmp_path / "gpu_mem.json"))
    from organism.distributed.gpu_worker_server import _GpuWorker

    w = _GpuWorker()
    out = w.pulse_batch([{"steps": 1}, {"steps": 1, "text": "ciao"}])
    assert out["count"] == 2
    saved = w.save_memory()
    assert saved["saved"] is True
    assert (tmp_path / "gpu_mem.json").exists()
    data = json.loads((tmp_path / "gpu_mem.json").read_text())
    assert "memory" in data
