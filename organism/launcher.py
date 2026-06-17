"""ORGANISM desktop launcher — entry point per .exe Windows."""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def _configure_paths() -> None:
    root = _bundle_root()
    static = root / "organism" / "nursery" / "static"
    if static.is_dir():
        os.environ.setdefault("ORGANISM_STATIC_DIR", str(static))
    data = Path.home() / ".organism"
    data.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ORGANISM_BABY_STATE", str(data / "baby_state.json"))
    os.environ.setdefault("ORGANISM_BIND", "127.0.0.1")
    os.environ.setdefault("ORGANISM_PORT", "8765")
    os.environ.setdefault("ORGANISM_IMPULSE", "1")
    os.environ.setdefault("ORGANISM_IMPULSE_DEVICE", "auto")
    os.environ.setdefault("ORGANISM_TEMPORAL", "1")
    os.environ.setdefault("ORGANISM_LOCAL_ONLY", "1")


def main() -> None:
    _configure_paths()
    port = int(os.environ.get("ORGANISM_PORT", "8765"))
    host = os.environ.get("ORGANISM_BIND", "127.0.0.1")
    url = f"http://{host}:{port}/"

    from organism.nursery.server import NurseryServer

    server = NurseryServer(host=host, port=port)

    def _run() -> None:
        server.start(open_browser=False)

    threading.Thread(target=_run, name="organism-server", daemon=True).start()

    for _ in range(40):
        time.sleep(0.25)
        try:
            import urllib.request

            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/baby/ready", timeout=1)
            break
        except Exception:
            pass

    print("\n  ORGANISM · Baby locale (GPU temporale)")
    print(f"  {url}")
    print("  Solo localhost — nessun server remoto.")
    print("  Motore Numba CUDA puro: python -m mindruntime.visualizer")
    print("  Chiudi questa finestra per uscire.\n")
    webbrowser.open(url)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nChiuso.")


if __name__ == "__main__":
    main()
