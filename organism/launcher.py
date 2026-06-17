"""ORGANISM desktop — native visualizer or nursery HTTP server."""

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
    os.environ.setdefault("ORGANISM_ENGINE", "v2")


def _run_nursery() -> int:
    """HTTP nursery on 127.0.0.1:8765 — used by Ink Admin plugin."""
    os.environ.setdefault("ORGANISM_NO_BROWSER", "1")
    # Strip subcommand so argparse in server.main() only sees flags.
    sys.argv = [sys.argv[0]]
    from organism.nursery.server import main as run_nursery

    run_nursery()
    return 0


def _run_visualizer() -> int:
    os.environ.setdefault("ORGANISM_NATIVE", "1")
    os.environ.setdefault("ORGANISM_NO_BROWSER", "1")
    from mindruntime.visualizer import main as run_native

    return int(run_native() or 0)


def main() -> None:
    _configure_paths()
    cmd = (sys.argv[1].lower() if len(sys.argv) > 1 else "").strip()
    if cmd in ("nursery", "serve", "server", "http", "baby"):
        raise SystemExit(_run_nursery())
    if cmd in ("viz", "visualizer", "native", ""):
        raise SystemExit(_run_visualizer())
    # Unknown arg — default nursery when spawned as sidecar from Ink Admin.
    if getattr(sys, "frozen", False) and cmd and not cmd.startswith("-"):
        raise SystemExit(_run_nursery())
    raise SystemExit(_run_visualizer())


if __name__ == "__main__":
    main()
