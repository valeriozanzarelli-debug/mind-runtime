"""Test report squadra Organism."""

from organism.reporting.squad_reporter import build_squad_report, persist_report


def test_build_squad_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "organism.reporting.squad_reporter.REPORT_DIR",
        tmp_path,
    )
    report = build_squad_report(
        baby_state={"neurons": 1500, "words_known": 42, "health": {"speech_diversity": 0.3}},
        tutor_state={"tutor": {"status": "running", "cycle": 10, "phase": "growing"}},
    )
    assert report["ok"] is True
    assert "Neuroni" in report["markdown"]
    assert len(report["gaps"]) >= 3
    path = persist_report(report)
    assert path.exists()
    assert (tmp_path / "latest.md").exists()
