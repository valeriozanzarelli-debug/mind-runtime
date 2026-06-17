"""ORGANISM desktop — cervello dendritico nativo, zero browser."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def _configure_paths() -> None:
    data = Path.home() / ".organism"
    data.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ORGANISM_DATA_DIR", str(data))
    os.environ.setdefault("ORGANISM_BABY_STATE", str(data / "baby_state.json"))
    os.environ.setdefault("ORGANISM_LOCAL_ONLY", "1")
    os.environ.setdefault("ORGANISM_NATIVE", "1")
    os.environ.setdefault("ORGANISM_NO_BROWSER", "1")


def main() -> None:
    """Avvia finestra OpenCV locale — nessun server HTTP, nessun browser."""
    _configure_paths()
    from mindruntime.visualizer import main as run_native

    raise SystemExit(run_native())


if __name__ == "__main__":
    main()
