"""Worker GPU locale — neuroni-pixel su NVIDIA (PC) per cervello delocalizzato."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Default risoluzione massima prudente per 8 GB VRAM
DEFAULT_W = int(os.environ.get("ORGANISM_IMPULSE_W", "4096"))
DEFAULT_H = int(os.environ.get("ORGANISM_IMPULSE_H", "3072"))


class _GpuWorker:
    def __init__(self) -> None:
        from organism.brain.impulse_scaffold import ImpulseScaffold
        from organism.brain.gpu_backend import gpu_info

        device = os.environ.get("ORGANISM_GPU_WORKER_DEVICE", "cuda")
        self.impulse = ImpulseScaffold(device=device, width=DEFAULT_W, height=DEFAULT_H)
        self.info = gpu_info()
        self.pulses = 0

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
        return {
            "ok": True,
            "pulses": self.pulses,
            "reading": reading.to_dict(),
            "stats": self.impulse.stats(),
            "themes": self.impulse.themes_for_speech(),
            "symbols": self.impulse.symbols_for_mind(),
            "workspace": self.impulse.workspace_overlay(),
        }


_worker: _GpuWorker | None = None


def get_worker() -> _GpuWorker:
    global _worker
    if _worker is None:
        _worker = _GpuWorker()
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
        if self.path.rstrip("/") in ("/health", "/api/gpu/health"):
            w = get_worker()
            self._json(
                200,
                {
                    "ok": True,
                    "device": w.info,
                    "resolution": f"{DEFAULT_W}x{DEFAULT_H}",
                    "pixels": DEFAULT_W * DEFAULT_H,
                    "pulses": w.pulses,
                },
            )
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in ("/pulse", "/api/gpu/pulse"):
            self._json(404, {"ok": False, "error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "invalid json"})
            return
        try:
            out = get_worker().pulse(payload)
            self._json(200, out)
        except Exception as exc:
            self._json(500, {"ok": False, "error": str(exc)[:200]})


def main() -> None:
    p = argparse.ArgumentParser(description="ORGANISM GPU worker — neuroni-pixel su PC locale")
    p.add_argument("--host", default=os.environ.get("ORGANISM_GPU_WORKER_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("ORGANISM_GPU_WORKER_PORT", "8770")))
    args = p.parse_args()
    print(f"GPU worker {DEFAULT_W}x{DEFAULT_H} = {DEFAULT_W*DEFAULT_H:,} neuroni-pixel")
    print(f"Listening http://{args.host}:{args.port}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
