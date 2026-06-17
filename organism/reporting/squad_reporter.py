"""Report squadra Haiku-style — metriche Baby + tutor per guidare i lavori Ops."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path.home() / ".organism" / "squad_reports"


def _ensure_dir() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR


def build_squad_report(*, baby_state: dict[str, Any], tutor_state: dict[str, Any]) -> dict[str, Any]:
    """Struttura report — l'agente Haiku in Ops Brain arricchisce via chat."""
    baby = baby_state if isinstance(baby_state, dict) else {}
    tutor = (tutor_state.get("tutor") if isinstance(tutor_state, dict) else {}) or {}
    health = baby.get("health") or baby
    now = datetime.now(timezone.utc).isoformat()

    neurons = health.get("neurons") or baby.get("neurons") or "?"
    synapses = health.get("synapses") or baby.get("synapses") or "?"
    words = health.get("words_known") or "?"
    diversity = health.get("speech_diversity") or "?"

    gaps = [
        "Pensiero continuo tra tick (non solo su stimolo caregiver)",
        "Memoria episodica richiamabile in dialogo",
        "Grounding visivo prima di narrare oggetti",
        "Benchmark fluency vs metriche tutor",
        "Insegnamento codice (fase 2)",
    ]

    next_tasks = [
        "Verificare tutor cycle + log in Ink Admin",
        "Patch minima mind-runtime su coscienza/memoria",
        "pytest + report Haiku a fine batch",
    ]

    markdown = "\n".join(
        [
            f"# Organism — report squadra · {now[:19]}Z",
            "",
            "## Stato Baby",
            f"- Neuroni: {neurons}",
            f"- Sinapsi: {synapses}",
            f"- Parole: {words}",
            f"- Diversità speech: {diversity}",
            "",
            "## Tutore",
            f"- Stato: {tutor.get('status', 'idle')}",
            f"- Fase: {tutor.get('phase', '—')}",
            f"- Cicli: {tutor.get('cycle', 0)}",
            f"- Intervallo: {tutor.get('interval_s', 45)}s",
            "",
            "## Gap noti",
            *[f"- {g}" for g in gaps],
            "",
            "## Prossimi passi suggeriti",
            *[f"- {t}" for t in next_tasks],
            "",
            "## Nota",
            "Organism NON sostituisce Ops Brain. Obiettivo: crescere cervello vero in locale.",
        ]
    )

    return {
        "ok": True,
        "generated_at": now,
        "markdown": markdown,
        "metrics": {
            "neurons": neurons,
            "synapses": synapses,
            "words_known": words,
            "speech_diversity": diversity,
            "tutor_status": tutor.get("status"),
            "tutor_cycle": tutor.get("cycle", 0),
        },
        "gaps": gaps,
        "next_tasks": next_tasks,
    }


def persist_report(report: dict[str, Any]) -> Path:
    _ensure_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"squad_report_{ts}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = REPORT_DIR / "latest.json"
    latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = REPORT_DIR / "latest.md"
    md_path.write_text(str(report.get("markdown", "")), encoding="utf-8")
    return path
