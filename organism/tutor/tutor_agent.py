"""Agente tutore — cresce Baby lentamente mentre il PC resta acceso.

Orchestra foundation → cicli integrati con pause lunghe, sonno notturno
e benchmark periodici. Pensato per girare accanto al nursery server.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

LOG_MAX = 200
DEFAULT_INTERVAL_S = float(os.environ.get("ORGANISM_TUTOR_INTERVAL", "45"))


def tutor_state_path() -> Path:
    data = Path(os.environ.get("ORGANISM_DATA_DIR", Path.home() / ".organism"))
    data.mkdir(parents=True, exist_ok=True)
    return data / "tutor_state.json"


@dataclass
class TutorState:
    status: str = "idle"  # idle | running | paused | error
    phase: str = "bootstrap"  # bootstrap | foundation | growing | sleeping
    cycle: int = 0
    foundation_done: bool = False
    started_at: float = 0.0
    last_tick_at: float = 0.0
    last_error: str = ""
    interval_s: float = DEFAULT_INTERVAL_S
    total_actions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TutorAgent:
    """Tutore autonomo — un passo alla volta, senza affogare Baby."""

    def __init__(
        self,
        *,
        baby_fn: Callable[[], Any],
        lock: threading.Lock | None = None,
        state_path: Path | None = None,
    ) -> None:
        self._baby_fn = baby_fn
        self._lock = lock or threading.Lock()
        self._state_path = state_path or tutor_state_path()
        self._state = TutorState()
        self._log: deque[str] = deque(maxlen=LOG_MAX)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()  # not paused
        self._load_state()

    @property
    def state(self) -> TutorState:
        return self._state

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._log.append(line)
        print(f"🎓 tutor {line}")

    def recent_log(self, n: int = 50) -> list[str]:
        return list(self._log)[-n:]

    def status_dict(self) -> dict[str, Any]:
        baby = self._baby_fn()
        baby_info: dict[str, Any] = {}
        try:
            with self._lock:
                affect = getattr(baby, "affect", None)
                presence = getattr(baby, "presence", None)
                baby_info = {
                    "born": bool(getattr(baby, "_born", False)),
                    "mood": getattr(affect, "label", lambda: "?")() if callable(getattr(affect, "label", None)) else str(getattr(presence, "mood_label", "?")),
                }
        except Exception:
            baby_info = {"born": False}
        return {
            "ok": True,
            "tutor": self._state.to_dict(),
            "baby": baby_info,
            "log_tail": self.recent_log(12),
        }

    def start(self, *, interval_s: float | None = None) -> dict[str, Any]:
        if interval_s is not None:
            self._state.interval_s = max(5.0, float(interval_s))
        if self._thread and self._thread.is_alive():
            self._pause.set()
            self._state.status = "running"
            self.log("Ripresa sessione tutore")
            self._persist_state()
            return self.status_dict()
        self._stop.clear()
        self._pause.set()
        self._state.status = "running"
        self._state.started_at = time.time()
        if self._state.started_at and not self._state.last_tick_at:
            self._state.last_tick_at = self._state.started_at
        self._thread = threading.Thread(target=self._loop, name="organism-tutor", daemon=True)
        self._thread.start()
        self.log(f"Avviato — intervallo {self._state.interval_s:.0f}s")
        self._persist_state()
        return self.status_dict()

    def pause(self) -> dict[str, Any]:
        self._pause.clear()
        self._state.status = "paused"
        self.log("In pausa")
        self._persist_state()
        return self.status_dict()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._pause.set()
        self._state.status = "idle"
        self.log("Fermato")
        self._persist_state()
        return self.status_dict()

    def tick_once(self) -> dict[str, Any]:
        """Un solo passo — utile per test o trigger manuale."""
        result = self._run_step()
        self._persist_state()
        return {"ok": True, "step": result, "tutor": self._state.to_dict()}

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._pause.is_set():
                time.sleep(1.0)
                continue
            try:
                self._run_step()
            except Exception as exc:
                self._state.status = "error"
                self._state.last_error = str(exc)[:200]
                self.log(f"ERRORE: {exc}")
            self._persist_state()
            # Attesa spezzata per stop/pause reattivi
            deadline = time.time() + self._state.interval_s
            while time.time() < deadline and not self._stop.is_set():
                if not self._pause.is_set():
                    break
                time.sleep(0.5)
        self._state.status = "idle"

    def _run_step(self) -> dict[str, Any]:
        baby = self._baby_fn()
        with self._lock:
            if not baby._born:
                self._state.phase = "bootstrap"
                self.log("Nascita Baby…")
                result = baby.birth()
                self._state.total_actions += 1
                self._state.last_tick_at = time.time()
                return {"action": "birth", "result": _brief(result)}

            if not self._state.foundation_done:
                self._state.phase = "foundation"
                self.log("Fondazione linguistica (lento, 1 ciclo)…")
                result = baby.train_foundation(repeats=1)
                self._state.foundation_done = True
                self._state.total_actions += 1
                self._state.last_tick_at = time.time()
                self.log("Fondazione completata — passo ai cicli integrati")
                return {"action": "foundation", "result": _brief(result)}

            self._state.phase = "growing"
            self._state.cycle += 1
            cycle = self._state.cycle
            self.log(f"Ciclo integrato #{cycle}")
            result = baby.train_integrated_cycle(cycle)
            self._state.total_actions += 1
            self._state.last_tick_at = time.time()

            if cycle % 50 == 0:
                self.log(f"Checkpoint #{cycle} — probe={result.get('probe', '?')}")

            if cycle % 200 == 0:
                self._state.phase = "sleeping"
                self.log("Sonno di consolidamento…")
                baby.sleep_cycle()
                self._state.phase = "growing"

            return {"action": "cycle", "cycle": cycle, "result": _brief(result)}

    def _load_state(self) -> None:
        path = self._state_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._state.cycle = int(data.get("cycle", 0))
            self._state.foundation_done = bool(data.get("foundation_done", False))
            self._state.interval_s = float(data.get("interval_s", DEFAULT_INTERVAL_S))
            self._state.total_actions = int(data.get("total_actions", 0))
            if data.get("log"):
                for line in data["log"][-LOG_MAX:]:
                    self._log.append(line)
        except Exception:
            pass

    def _persist_state(self) -> None:
        try:
            payload = {
                **self._state.to_dict(),
                "log": list(self._log)[-LOG_MAX:],
            }
            self._state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


def _brief(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"value": str(result)[:120]}
    out: dict[str, Any] = {}
    for k in ("ok", "cycle", "probe", "spoke_words", "taught", "pruned", "stats"):
        if k in result:
            v = result[k]
            if isinstance(v, dict):
                out[k] = {kk: v[kk] for kk in list(v)[:4]}
            else:
                out[k] = v
    if not out:
        out["keys"] = list(result.keys())[:6]
    return out
