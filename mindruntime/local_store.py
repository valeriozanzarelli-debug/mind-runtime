"""Persistenza locale — nessun browser, nessun server HTTP richiesto.

Salva in %USERPROFILE%\\.organism\\mindruntime\\
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def store_dir() -> Path:
    root = Path(os.environ.get("ORGANISM_DATA_DIR", Path.home() / ".organism"))
    d = root / "mindruntime"
    d.mkdir(parents=True, exist_ok=True)
    return d


def journal_path() -> Path:
    return store_dir() / "session.jsonl"


def state_path() -> Path:
    return store_dir() / "state_latest.json"


def append_tick(record: dict[str, Any]) -> None:
    """Append una riga al journal di sessione."""
    rec = {"ts": time.time(), **record}
    with journal_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_snapshot(
    engine: Any,
    *,
    frames: int,
    fps: float,
) -> Path:
    """Salva stato corrente + metriche su disco locale."""
    stats = engine.stats
    export = engine.export_state_for_training()
    payload: dict[str, Any] = {
        "saved_at": time.time(),
        "frames": frames,
        "fps": round(fps, 2),
        "tick": stats.tick,
        "backend": stats.backend,
        "order_parameter": stats.order_parameter,
        "conscious": stats.conscious,
        "phase_transition": stats.phase_transition,
        "mean_coherence": stats.mean_coherence,
        "avalanche": stats.avalanche,
        "soc_coupling": stats.soc_coupling,
        "lock_in": stats.lock_in,
        "locked_symbol": stats.locked_symbol,
        "recognition": [{"symbol": s, "score": sc} for s, sc in stats.last_recognition],
        "arrays": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in export.items()
        },
    }
    path = state_path()
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_last_state() -> dict[str, Any] | None:
    p = state_path()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def store_info() -> dict[str, str]:
    d = store_dir()
    return {
        "dir": str(d),
        "journal": str(journal_path()),
        "state": str(state_path()),
    }
