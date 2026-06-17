"""Client GPU remoto — server (RAM) delega neuroni-pixel al PC locale (CUDA)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from organism.brain.impulse_consciousness import ConsciousnessReading
from organism.cognition.workspace import WorkspaceState


def remote_gpu_url() -> str | None:
    url = os.environ.get("ORGANISM_GPU_REMOTE", "").strip()
    return url.rstrip("/") if url else None


class RemoteImpulseScaffold:
    """Proxy verso gpu_worker_server sul PC con NVIDIA."""

    def __init__(self, base_url: str, *, timeout_s: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._last_reading: ConsciousnessReading | None = None
        self._pulse_count = 0
        self._pending_gray: list | None = None
        self._pending_text: str | None = None
        self._pending_audio: list[float] | None = None
        self._last_stats: dict[str, Any] = {"remote": True, "url": self.base_url}

    @property
    def last_reading(self) -> ConsciousnessReading | None:
        return self._last_reading

    def perceive_visual(self, gray: list | None, *, gain: float = 0.9) -> None:
        self._pending_gray = gray
        self._gain = gain

    def perceive_audio(self, bands: list[float] | None) -> None:
        self._pending_audio = bands

    def perceive_text(self, text: str | None) -> None:
        self._pending_text = text

    def pulse(self, *, steps: int = 2) -> ConsciousnessReading:
        payload: dict[str, Any] = {"steps": steps}
        if self._pending_gray:
            payload["gray"] = self._pending_gray
            payload["gain"] = getattr(self, "_gain", 0.9)
        if self._pending_text:
            payload["text"] = self._pending_text
        if self._pending_audio:
            payload["audio_bands"] = self._pending_audio

        data = self._post("/pulse", payload)
        self._pulse_count += 1
        self._pending_gray = None
        self._pending_text = None
        self._pending_audio = None

        reading_d = data.get("reading", {})
        fp = reading_d.get("focus_point")
        self._last_reading = ConsciousnessReading(
            conscious=bool(reading_d.get("conscious")),
            ignition=float(reading_d.get("ignition", 0)),
            mode=str(reading_d.get("mode", "flow")),  # type: ignore[arg-type]
            focus_region=str(reading_d.get("focus_region", "")),
            focus_point=tuple(fp) if fp else None,
            sensations=list(reading_d.get("sensations", [])),
            memories_recalled=list(reading_d.get("memories_recalled", [])),
            thoughts=list(reading_d.get("thoughts", [])),
            broadcast=list(reading_d.get("broadcast", [])),
            blobs=list(reading_d.get("blobs", [])),
            regions=list(reading_d.get("regions", [])),
            flux=float(reading_d.get("flux", 0)),
            novelty=float(reading_d.get("novelty", 0)),
            self_signal=float(reading_d.get("self_signal", 0)),
            recognized=list(reading_d.get("recognized", [])),
            phase_coherence=float(reading_d.get("phase_coherence", 0)),
            acceleration=float(reading_d.get("acceleration", 0)),
        )
        self._last_stats = data.get("stats", self._last_stats)
        self._last_themes = list(data.get("themes", []))
        self._last_symbols = list(data.get("symbols", []))
        self._last_workspace = dict(data.get("workspace", {}))
        return self._last_reading

    def themes_for_speech(self) -> list[str]:
        return list(getattr(self, "_last_themes", []))

    def symbols_for_mind(self) -> list[str]:
        return list(getattr(self, "_last_symbols", []))

    def workspace_overlay(self) -> dict[str, Any]:
        return dict(getattr(self, "_last_workspace", {}))

    def stats(self) -> dict[str, Any]:
        base = dict(self._last_stats)
        base["pulses"] = self._pulse_count
        base["remote"] = True
        base["url"] = self.base_url
        if self._last_reading:
            base["consciousness"] = self._last_reading.to_dict()
        return base

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GPU remoto non raggiungibile ({url}): {exc}") from exc


def merge_remote_workspace(ws: WorkspaceState, impulse: RemoteImpulseScaffold | None) -> WorkspaceState:
    if impulse is None or impulse.last_reading is None:
        return ws
    ov = impulse.workspace_overlay()
    if not ov:
        return ws
    ws.ignition = max(ws.ignition, float(ov.get("ignition", 0)) * 0.92)
    ws.conscious = ws.conscious or bool(ov.get("conscious"))
    merged = list(dict.fromkeys(list(ov.get("broadcast", [])) + list(ws.broadcast)))
    ws.broadcast = merged[:12]
    if ov.get("focus"):
        ws.focus = str(ov["focus"])
    mode = ov.get("mode")
    if mode in ("speak", "reflect", "flow"):
        ws.mode = mode  # type: ignore[assignment]
    ws.self_signal = max(ws.self_signal, float(ov.get("self_signal", 0)))
    return ws
