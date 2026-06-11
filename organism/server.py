"""Nursery server — Baby (default) + Lab dashboard."""

from __future__ import annotations

import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from organism.autonomous.baby_agent import BabyAgent
from organism.cognition.brain_pulse import BrainPulse
from organism.nursery.nursery import Nursery
from organism.nursery.phases import phases_dict

STATIC_DIR = Path(__file__).parent / "static"


class NurseryServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        seed: int = 42,
        *,
        base_path: str = "",
        public_url: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.base_path = _norm_base(base_path or os.environ.get("ORGANISM_BASE_PATH", ""))
        self.public_url = (public_url or os.environ.get("ORGANISM_PUBLIC_URL", "")).rstrip("/")
        self.nursery = Nursery(seed=seed)
        self.baby = BabyAgent(seed=seed)
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        interval = float(os.environ.get("ORGANISM_PULSE_INTERVAL", "1.2"))
        self._brain_pulse = BrainPulse(self._pulse_tick, interval_s=interval)

    def _pulse_tick(self) -> dict:
        with self._lock:
            if not self.baby._born:
                return {"alive": False}
            n = self.baby.affect._pulse_count
            return self.baby.brain_pulse_tick(persist=n > 0 and n % 25 == 0)

    def start(self, *, open_browser: bool = False) -> None:
        self._brain_pulse.start()
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        local = f"http://{self.host}:{self.port}{self.base_path or '/'}"
        public = self.public_url or local
        print(f"\n🧬 ORGANISM Baby")
        print(f"   Locale:  {local}")
        if self.public_url:
            print(f"   Pubblico: {public}")
        print(f"   Lab:     {public.rstrip('/')}/lab")
        print("   Ctrl+C per uscire\n")
        if open_browser:
            import webbrowser
            webbrowser.open(public if self.public_url else local)
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            print("\nChiuso.")
            self._server.shutdown()

    def _with_lock(self, fn):
        with self._lock:
            return fn()


def _norm_base(path: str) -> str:
    if not path or path == "/":
        return ""
    return "/" + path.strip("/")


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

        def _route_path(self) -> str:
            path = urlparse(self.path).path
            bp = server.base_path
            if bp and path.startswith(bp):
                path = path[len(bp) :] or "/"
            return path

        def do_GET(self) -> None:
            path = self._route_path()
            if path in ("/", "/baby", "/baby.html"):
                return _serve_baby_html(self, server.base_path)
            if path in ("/lab", "/index.html"):
                return _serve_file(self, STATIC_DIR / "index.html")
            if path.startswith("/static/"):
                return _serve_file(self, STATIC_DIR / path[8:])
            if path == "/api/baby/state":
                def _state():
                    s = server.baby.state()
                    s["brain_pulse"] = server._brain_pulse.stats()
                    return s

                return _json(self, server._with_lock(_state))
            if path == "/api/baby/debug/thoughts":
                return _json(
                    self,
                    {"thoughts": server._with_lock(lambda: server.baby.debug_thoughts(25))},
                )
            if path == "/api/baby/health":
                return _json(self, server._with_lock(server.baby.health))
            if path == "/api/baby/consciousness":
                def _consciousness():
                    qs = parse_qs(urlparse(self.path).query)
                    try:
                        limit = min(120, max(8, int(qs.get("n", ["48"])[0])))
                    except (ValueError, IndexError):
                        limit = 48
                    return {
                        "stream": server.baby.consciousness_recent(limit),
                        "last_moment": server.baby._last_moment.to_dict() if server.baby._last_moment else None,
                    }

                return _json(self, server._with_lock(_consciousness))
            if path == "/api/config":
                return _json(
                    self,
                    {"base_path": server.base_path, "public_url": server.public_url},
                )
            if path == "/api/state":
                return _json(self, server._with_lock(server.nursery.state))
            if path == "/api/phases":
                return _json(self, {"phases": phases_dict()})
            if path == "/api/verify":
                return _json(self, server._with_lock(server.nursery.verify))
            if path == "/api/graph":
                org = server.nursery.org
                g = org.brain.export_active_subgraph() if org else {"nodes": [], "edges": []}
                return _json(self, g)
            self.send_error(404)

        def do_POST(self) -> None:
            path = self._route_path()
            body = _read_json(self)

            if path == "/api/baby/birth":
                return _json(self, server._with_lock(server.baby.birth))
            if path == "/api/baby/rebirth":
                if os.environ.get("ORGANISM_ALLOW_REBIRTH", "0") != "1":
                    return _json(
                        self,
                        {
                            "error": "rebirth_disabled",
                            "message": "Rebirth disabilitato — memoria preservata",
                        },
                        status=403,
                    )
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.rebirth(
                            keep_dialogue=bool(body.get("keep_dialogue", True)),
                        )
                    ),
                )
            if path == "/api/baby/sense":
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.sense(
                            image_gray=body.get("image_gray"),
                            image_rgba=body.get("image_rgba"),
                            image_b64=body.get("image_b64"),
                            image_w=int(body.get("image_w", 64)),
                            image_h=int(body.get("image_h", 64)),
                            audio_b64=body.get("audio_b64"),
                            text=body.get("text"),
                            color_rgb=body.get("color_rgb"),
                        )
                    ),
                )
            if path == "/api/baby/autonomous":
                return _json(self, server._with_lock(server.baby.autonomous_tick))
            if path == "/api/baby/flow":
                return _json(self, server._with_lock(server.baby.flow))
            if path == "/api/baby/look":
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.look(
                            image_gray=body.get("image_gray"),
                            image_b64=body.get("image_b64"),
                            image_w=int(body.get("image_w", 64)),
                            image_h=int(body.get("image_h", 64)),
                            color_rgb=body.get("color_rgb"),
                            image_rgba=body.get("image_rgba"),
                        )
                    ),
                )
            if path == "/api/baby/glance":
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.glance(
                            image_gray=body.get("image_gray"),
                            image_b64=body.get("image_b64"),
                            image_w=int(body.get("image_w", 64)),
                            image_h=int(body.get("image_h", 64)),
                            color_rgb=body.get("color_rgb"),
                            image_rgba=body.get("image_rgba"),
                            min_interval_s=float(body.get("min_interval_s", 10)),
                        )
                    ),
                )
            if path == "/api/baby/hear":
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.hear_spoken(
                            body.get("phrase", ""),
                            teach_focus=bool(body.get("teach_focus", False)),
                            source=str(body.get("source", "caregiver")),
                            image_gray=body.get("image_gray"),
                            image_b64=body.get("image_b64"),
                            image_w=int(body.get("image_w", 64)),
                            image_h=int(body.get("image_h", 64)),
                            color_rgb=body.get("color_rgb"),
                            image_rgba=body.get("image_rgba"),
                        )
                    ),
                )
            if path == "/api/baby/teach-attention":
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.teach_attention(
                            body.get("phrase", ""),
                            image_gray=body.get("image_gray"),
                            image_b64=body.get("image_b64"),
                            image_w=int(body.get("image_w", 64)),
                            image_h=int(body.get("image_h", 64)),
                            color_rgb=body.get("color_rgb"),
                            image_rgba=body.get("image_rgba"),
                        )
                    ),
                )
            if path == "/api/baby/teach-object":
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.teach_object(
                            body.get("name", ""),
                            image_gray=body.get("image_gray"),
                            image_b64=body.get("image_b64"),
                            image_w=int(body.get("image_w", 64)),
                            image_h=int(body.get("image_h", 64)),
                            color_rgb=body.get("color_rgb"),
                            image_rgba=body.get("image_rgba"),
                            phrase=body.get("phrase"),
                        )
                    ),
                )
            if path == "/api/baby/teach":
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.teach_repetition(
                            body.get("phrase", ""),
                            stimulus_key=body.get("stimulus_key"),
                            image_gray=body.get("image_gray"),
                            image_b64=body.get("image_b64"),
                            image_w=int(body.get("image_w", 64)),
                            image_h=int(body.get("image_h", 64)),
                        )
                    ),
                )
            if path == "/api/baby/teach-dialogue":
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.teach_dialogue(
                            body.get("when", ""),
                            body.get("say", ""),
                            kind=body.get("kind", "speech"),
                        )
                    ),
                )
            if path == "/api/baby/absorb":
                texts = body.get("texts") or body.get("text", "")
                if isinstance(texts, str):
                    texts = [texts] if texts else []
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.absorb_vocabulary(
                            list(texts),
                            boost=float(body.get("boost", 1.0)),
                        )
                    ),
                )
            if path == "/api/baby/research":
                return _json(self, server._with_lock(server.baby.research_once))
            if path == "/api/baby/reflect":
                prompt = body.get("prompt", "")
                return _json(
                    self,
                    server._with_lock(lambda: server.baby.reflect(prompt=prompt)),
                )
            if path == "/api/baby/self-hear":
                return _json(
                    self,
                    server._with_lock(lambda: server.baby.self_hear(text=body.get("text", ""))),
                )
            if path == "/api/baby/read":
                return _json(
                    self,
                    server._with_lock(lambda: server.baby.read(body.get("text", ""))),
                )
            if path == "/api/baby/task":
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.baby.assign_task(
                            body.get("kind", "repeat"),
                            body.get("prompt", ""),
                            body.get("target", ""),
                        )
                    ),
                )

            if path == "/api/birth":
                return _json(self, server._with_lock(server.nursery.birth))
            if path == "/api/teach":
                data = body.get("input", body)
                kwargs = {k: v for k, v in body.items() if k not in ("input", "modality", "phase")}
                return _json(
                    self,
                    server._with_lock(
                        lambda: server.nursery.teach(
                            data,
                            phase=body.get("phase"),
                            modality=body.get("modality", "speech"),
                            **kwargs,
                        )
                    ),
                )
            if path.startswith("/api/phase/"):
                phase_id = path.split("/")[-1]
                if body.get("lesson"):
                    return _json(
                        self,
                        server._with_lock(lambda: server.nursery.run_lesson(phase_id, body["lesson"])),
                    )
                return _json(self, server._with_lock(lambda: server.nursery.run_phase(phase_id)))
            if path == "/api/curriculum":
                return _json(self, server._with_lock(server.nursery.run_full_curriculum))
            if path == "/api/sleep":
                return _json(self, server._with_lock(server.nursery.sleep))
            self.send_error(404)

    return Handler


def _serve_baby_html(handler: BaseHTTPRequestHandler, base_path: str) -> None:
    raw = (STATIC_DIR / "baby.html").read_text(encoding="utf-8")
    content = raw.replace("__ORGANISM_BASE__", base_path).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    handler.wfile.write(content)


def _serve_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.exists():
        handler.send_error(404)
        return
    content = path.read_bytes()
    ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    handler.wfile.write(content)


def _json(handler: BaseHTTPRequestHandler, data: Any, *, status: int = 200) -> None:
    payload = json.dumps(data, ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length).decode())


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="ORGANISM Baby — nasce, impara, parla")
    p.add_argument("--host", default=os.environ.get("ORGANISM_BIND", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("ORGANISM_PORT", "8765")))
    p.add_argument("--base-path", default=os.environ.get("ORGANISM_BASE_PATH", ""))
    p.add_argument("--public-url", default=os.environ.get("ORGANISM_PUBLIC_URL", ""))
    p.add_argument("--browser", action="store_true")
    args = p.parse_args()
    NurseryServer(
        args.host,
        args.port,
        base_path=args.base_path,
        public_url=args.public_url or None,
    ).start(open_browser=args.browser)


if __name__ == "__main__":
    main()
