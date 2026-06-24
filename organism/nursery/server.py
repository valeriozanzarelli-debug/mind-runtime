"""Nursery HTTP server — compatible with Ink Admin organism client."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from organism.brain.runtime import BrainRuntime


class NurseryServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, seed: int = 42) -> None:
        self.host = host
        self.port = port
        self.brain = BrainRuntime.create(seed=seed, use_gpu=os.environ.get("ORGANISM_GPU", "0") == "1")
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._pulse_thread: threading.Thread | None = None
        self._pulse_stop = threading.Event()
        interval = float(os.environ.get("ORGANISM_PULSE_INTERVAL", "0.8"))

        def _pulse_loop() -> None:
            while not self._pulse_stop.is_set():
                with self._lock:
                    if self.brain._born:
                        self.brain.brain_pulse_tick()
                self._pulse_stop.wait(interval)

        self._pulse_thread = threading.Thread(target=_pulse_loop, name="brain-pulse", daemon=True)
        self._pulse_thread.start()

    def start(self) -> None:
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        print(f"\n🧠 ORGANISM Biological Brain")
        print(f"   http://{self.host}:{self.port}/")
        print(f"   Neuroni target: 23_800 | Ctrl+C per uscire\n")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            print("\nChiuso.")
            self._pulse_stop.set()
            self._server.shutdown()

    def _with_lock(self, fn):
        try:
            with self._lock:
                return fn()
        except Exception as exc:
            return {"error": str(exc), "ok": False}


def _make_handler(server: NurseryServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/baby/health":
                return _json(self, server._with_lock(server.brain.health))
            if path == "/api/baby/ready":
                return _json(self, {"born": server.brain._born, "ok": True})
            if path == "/api/baby/state":
                qs = parse_qs(urlparse(self.path).query)
                lite = qs.get("lite", ["1"])[0].lower() in ("1", "true", "yes")
                def _state():
                    s = server.brain.state_lite() if lite else server.brain.state_lite()
                    s["brain_pulse"] = {"last": {"pulses": server.brain._pulse_count}}
                    return s
                return _json(self, server._with_lock(_state))
            if path == "/api/baby/impulse":
                def _impulse():
                    cap = server.brain.gpu.estimate_capacity_for_brain(
                        server.brain.brain.neuron_count,
                        server.brain.brain.synapse_count,
                    )
                    return {
                        "enabled": True,
                        "stats": {
                            "backend": cap.backend,
                            "device": cap.device,
                            "flux": cap.ticks_per_second,
                            "hybrid_mode": cap.backend,
                            "width": server.brain.brain.neuron_count,
                            "pixels": server.brain.brain.neuron_count,
                        },
                        "energy_w": server.brain.brain.neuron_count,
                        "energy_h": 1,
                        "energy_d": 1,
                        "spatial": "1d",
                        "reading": {
                            "conscious": server.brain.consciousness.ignition,
                            "ignition": server.brain.consciousness.phi,
                            "focus_region": server.brain.consciousness.focus_region,
                        },
                    }
                return _json(self, server._with_lock(_impulse))
            if path == "/api/baby/consciousness":
                qs = parse_qs(urlparse(self.path).query)
                try:
                    limit = min(120, max(8, int(qs.get("n", ["48"])[0])))
                except (ValueError, IndexError):
                    limit = 48
                try:
                    since_seq = max(0, int(qs.get("since", ["0"])[0]))
                except (ValueError, IndexError):
                    since_seq = 0
                def _consciousness():
                    ev = server.brain.consciousness_events(since_seq=since_seq, limit=limit)
                    return {
                        "stream": server.brain.consciousness_recent(limit),
                        **ev,
                    }
                return _json(self, server._with_lock(_consciousness))
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            body = _read_json(self)
            if path == "/api/baby/birth":
                return _json(self, server._with_lock(server.brain.birth))
            if path == "/api/baby/wake":
                def _wake():
                    if not server.brain._born:
                        server.brain.birth()
                    return {"awake": True}
                return _json(self, server._with_lock(_wake))
            if path == "/api/baby/chat":
                text = str(body.get("text", ""))
                return _json(self, server._with_lock(lambda: server.brain.chat(text)))
            if path == "/api/baby/hear":
                phrase = str(body.get("phrase", body.get("text", "")))
                def _hear():
                    server.brain.perceive_text(phrase)
                    result = server.brain.tick()
                    return {"heard": phrase, "tick": result}
                return _json(self, server._with_lock(_hear))
            self.send_response(404)
            self.end_headers()

    return Handler


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}


def _json(handler: BaseHTTPRequestHandler, data: Any, *, status: int = 200) -> None:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def main() -> None:
    host = os.environ.get("ORGANISM_HOST", "127.0.0.1")
    port = int(os.environ.get("ORGANISM_PORT", "8765"))
    seed = int(os.environ.get("ORGANISM_SEED", "42"))
    NurseryServer(host=host, port=port, seed=seed).start()


if __name__ == "__main__":
    main()
