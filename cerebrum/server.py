"""Server HTTP locale di CEREBRUM.

Espone il cervello su 127.0.0.1:8788 (default) con endpoint compatibili con
Ink Admin (`/health`, `/ready`, `/status`, `/chat`, `/train`) più endpoint di
introspezione per 'vedere dentro' il cervello (`/introspect`, `/consciousness`,
`/see`, `/hear`, `/care`, `/power`).

Solo localhost: nessun ascolto su interfacce pubbliche. Il canale verso il
server remoto (per usarlo/consultarlo) è gestito da Ink Admin, non da qui.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from cerebrum import DEFAULT_PORT, __version__
from cerebrum.brain import BrainConfig, Cerebrum


class _Handler(BaseHTTPRequestHandler):
    brain: Cerebrum = None  # iniettato dal server

    def log_message(self, *args):  # silenzia il log di default
        pass

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_OPTIONS(self):
        self._send(200, {"ok": True})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        q = parse_qs(parsed.query)
        b = self.brain
        if path == "/health":
            self._send(200, b.health())
        elif path == "/ready":
            self._send(200, {"ready": b.alive, "alive": b.alive})
        elif path == "/status":
            self._send(200, b.status())
        elif path == "/introspect":
            self._send(200, b.introspect())
        elif path == "/consciousness":
            since = int(q.get("since", [0])[0])
            n = int(q.get("n", [50])[0])
            self._send(200, b.consciousness(since=since, n=n))
        elif path in ("/", "/info"):
            self._send(200, {"name": "CEREBRUM", "version": __version__,
                             "endpoints": ["/health", "/ready", "/status", "/introspect",
                                           "/consciousness", "/chat", "/train", "/hear",
                                           "/see", "/care", "/power"]})
        else:
            self._send(404, {"error": "not found", "path": path})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        b = self.brain
        data = self._body()
        if path == "/chat":
            text = str(data.get("text", "")).strip()
            if not text:
                self._send(400, {"error": "text richiesto"})
                return
            self._send(200, b.respond(text))
        elif path == "/hear":
            self._send(200, b.hear(str(data.get("text", ""))))
        elif path == "/see":
            self._send(200, b.see(frame=data.get("frame"), stats=data.get("stats")))
        elif path == "/care":
            self._send(200, b.care(str(data.get("action", ""))))
        elif path == "/train":
            # 'training' qui = esposizione a stimoli/esperienza guidata
            steps = int(data.get("steps", 80))
            for _ in range(max(1, min(steps, 2000))):
                b.field.step(None, b.chem.as_dict())
            self._send(200, {"ok": True, "steps": steps,
                             "metrics": {"phi": b.field.phi_estimate()}})
        elif path == "/power":
            action = str(data.get("action", "")).lower()
            if action in ("on", "start", "birth"):
                b.birth()
                self._send(200, {"ok": True, "state": "on", "alive": b.alive})
            elif action in ("off", "stop", "shutdown"):
                b.shutdown()
                self._send(200, {"ok": True, "state": "off", "alive": b.alive})
            else:
                self._send(400, {"error": "action deve essere on/off"})
        else:
            self._send(404, {"error": "not found", "path": path})


def serve(port: int = DEFAULT_PORT, neurons: int = 4096) -> None:
    vault = os.path.join(
        os.environ.get("CEREBRUM_HOME", os.path.expanduser("~/.cerebrum")),
        "memory.json",
    )
    brain = Cerebrum(BrainConfig(neurons=neurons, vault_path=vault))
    brain.birth()
    _Handler.brain = brain

    host = os.environ.get("CEREBRUM_HOST", "127.0.0.1")
    httpd = ThreadingHTTPServer((host, port), _Handler)
    backend = brain.field.backend
    print(f"CEREBRUM v{__version__} — {backend['engine']}/{backend['device']}"
          + (f" ({backend.get('gpu')})" if backend.get("gpu") else ""))
    print(f"In ascolto su http://{host}:{port}  ({brain.field.neurons} neuroni, "
          f"{brain.field.computational_units} unità)")
    print("Cervello acceso e in pensiero continuo. Ctrl+C per spegnere.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nSpegnimento…")
    finally:
        brain.shutdown()
        httpd.server_close()
