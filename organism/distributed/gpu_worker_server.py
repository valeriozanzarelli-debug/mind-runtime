"""Worker GPU locale — neuroni-pixel su NVIDIA (PC) per cervello delocalizzato."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Default 3D — 512×384×128 ≈ 25M voxel-neuroni su 8 GB VRAM
DEFAULT_W = int(os.environ.get("ORGANISM_IMPULSE_W", "512"))
DEFAULT_H = int(os.environ.get("ORGANISM_IMPULSE_H", "384"))
DEFAULT_D = int(os.environ.get("ORGANISM_IMPULSE_D", "128"))


def memory_path() -> Path:
    raw = os.environ.get("ORGANISM_GPU_MEMORY", "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".organism" / "gpu_impulse_memory.json"


class _GpuWorker:
    def __init__(self) -> None:
        from organism.brain.impulse_scaffold import ImpulseScaffold
        from organism.brain.gpu_backend import gpu_info

        device = os.environ.get("ORGANISM_GPU_WORKER_DEVICE", "cuda")
        self.impulse = ImpulseScaffold(device=device, width=DEFAULT_W, height=DEFAULT_H, depth=DEFAULT_D)
        self.info = gpu_info()
        self.pulses = 0
        self._memory_file = memory_path()
        self._load_memory()

    def _load_memory(self) -> None:
        if not self._memory_file.exists():
            return
        try:
            data = json.loads(self._memory_file.read_text(encoding="utf-8"))
            self.impulse.load_dict(data)
        except (json.JSONDecodeError, OSError):
            pass

    def save_memory(self) -> dict[str, Any]:
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        payload = self.impulse.to_dict()
        payload["pulses"] = self.pulses
        self._memory_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"saved": True, "path": str(self._memory_file), "episodes": payload.get("memory", {}).get("stats", {})}

    def pulse(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("gray"):
            self.impulse.perceive_visual(payload["gray"], gain=float(payload.get("gain", 0.9)))
        if payload.get("text"):
            self.impulse.perceive_text(str(payload["text"]))
        if payload.get("audio_bands"):
            self.impulse.perceive_audio(list(payload["audio_bands"]))
        steps = int(payload.get("steps", 2))
        reading = self.impulse.pulse(steps=steps)
        self.pulses += 1
        if self.pulses % 50 == 0:
            self.save_memory()
        return {
            "ok": True,
            "pulses": self.pulses,
            "reading": reading.to_dict(),
            "stats": self.impulse.stats(),
            "themes": self.impulse.themes_for_speech(),
            "symbols": self.impulse.symbols_for_mind(),
            "workspace": self.impulse.workspace_overlay(),
        }

    def pulse_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        readings: list[dict[str, Any]] = []
        for item in items[:32]:
            readings.append(self.pulse(item))
        return {"ok": True, "count": len(readings), "readings": readings, "pulses": self.pulses}


_worker: _GpuWorker | None = None


def get_worker() -> _GpuWorker:
    global _worker
    if _worker is None:
        _worker = _GpuWorker()
        atexit.register(lambda: _worker.save_memory() if _worker else None)
    return _worker


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _json(self, code: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = self.path.rstrip("/")
        if path in ("/health", "/api/gpu/health"):
            w = get_worker()
            self._json(
                200,
                {
                    "ok": True,
                    "device": w.info,
                    "resolution": f"{DEFAULT_W}x{DEFAULT_H}x{DEFAULT_D}",
                    "pixels": DEFAULT_W * DEFAULT_H * DEFAULT_D,
                    "spatial": "3d" if DEFAULT_D >= 8 else "2d",
                    "pulses": w.pulses,
                    "memory_path": str(w._memory_file),
                },
            )
            return
        if path in ("/memory", "/api/gpu/memory"):
            w = get_worker()
            self._json(200, {"ok": True, "memory": w.impulse.memory.to_dict(), "pulses": w.pulses})
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "invalid json"})
            return

        if path in ("/pulse", "/api/gpu/pulse"):
            try:
                out = get_worker().pulse(payload)
                self._json(200, out)
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)[:200]})
            return

        if path in ("/pulse/batch", "/api/gpu/pulse/batch"):
            items = payload.get("items", [])
            if not isinstance(items, list):
                self._json(400, {"ok": False, "error": "items must be a list"})
                return
            try:
                out = get_worker().pulse_batch(items)
                self._json(200, out)
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)[:200]})
            return

        if path in ("/memory/save", "/api/gpu/memory/save"):
            try:
                out = get_worker().save_memory()
                self._json(200, {"ok": True, **out})
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)[:200]})
            return

        self._json(404, {"ok": False, "error": "not found"})


def main() -> None:
    p = argparse.ArgumentParser(description="ORGANISM GPU worker — neuroni-pixel su PC locale")
    p.add_argument("--host", default=os.environ.get("ORGANISM_GPU_WORKER_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("ORGANISM_GPU_WORKER_PORT", "8770")))
    args = p.parse_args()
    print(f"GPU worker {DEFAULT_W}x{DEFAULT_H}x{DEFAULT_D} = {DEFAULT_W*DEFAULT_H*DEFAULT_D:,} voxel-neuroni")
    print(f"Listening http://{args.host}:{args.port}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
