"""Flusso di coscienza leggibile — pensieri interni anche quando tace."""

from __future__ import annotations

import re
import time
from typing import Any


def is_question(text: str) -> bool:
    t = text.strip().lower().rstrip("?")
    if not t:
        return False
    starters = (
        "chi",
        "come",
        "cosa",
        "che ",
        "dove",
        "quando",
        "perché",
        "perche",
        "quale",
        "quanto",
        "quanti",
    )
    return text.strip().endswith("?") or any(
        t == s.rstrip() or t.startswith(s) for s in starters
    )


def build_consciousness_stream(
    *,
    heard: str | None = None,
    thought: dict[str, Any] | None = None,
    workspace: dict[str, Any] | None = None,
    self_state: dict[str, Any] | None = None,
    spoke: str = "",
    wants_voice: bool = False,
    presence: dict[str, Any] | None = None,
    motor_will: bool = False,
    wave: dict[str, Any] | None = None,
    emotion: dict[str, Any] | None = None,
    phase: str = "sense",
    taught_anchor: str = "",
) -> list[str]:
    """Linee in italiano — letteralmente cosa sta succedendo dentro."""
    lines: list[str] = []
    th = thought or {}
    ws = workspace or {}
    pres = presence or {}
    emo = emotion or {}

    if phase:
        lines.append(f"[{phase}]")

    if heard:
        q = " (domanda)" if is_question(heard) else ""
        lines.append(f"udito{q}: «{heard[:100]}»")

    if wave and wave.get("label"):
        lines.append(f"onda: {wave['label']}")

    ignition = float(ws.get("ignition", 0))
    conscious = bool(ws.get("conscious"))
    mode = ws.get("mode", "flow")
    focus = ws.get("focus", "")
    self_sig = float(ws.get("self_signal", 0))

    if conscious:
        lines.append(f"coscienza accesa ({ignition:.2f}) · modo {mode}")
    else:
        lines.append(f"pre-coscienza ({ignition:.2f}) · modo {mode}")
    if focus and focus != "FOCUS:sub":
        lines.append(f"attenzione: {focus[:60]}")

    themes = [t for t in th.get("themes", []) if len(t) > 1][:10]
    if themes:
        lines.append(f"pensa: {' · '.join(themes)}")
    pressure = float(th.get("pressure", 0))
    if pressure > 0.05:
        lines.append(f"pressione mentale: {pressure:.2f}")

    for sym in th.get("symbols", []):
        if sym.startswith("BRAIN:") or sym.startswith("CURIOSITY:") or sym.startswith("NOVELTY:"):
            continue
        lines.append(f"  ↳ {sym[:72]}")

    for b in ws.get("broadcast", [])[:5]:
        if b not in themes:
            lines.append(f"  ◉ {b[:48]}")

    self_stream = (self_state or {}).get("stream", [])
    if self_stream:
        lines.append(f"sé: {' · '.join(self_stream[:6])}")
    if self_sig > 0.05:
        lines.append(f"percepisce sé ({self_sig:.2f})")

    if emo.get("label"):
        lines.append(f"sentimento: {emo['label']}")

    if taught_anchor:
        lines.append(f"richiama memoria (non copia): «{taught_anchor[:60]}»")

    if spoke:
        lines.append(f"esprime ad alta voce: «{spoke[:120]}»")
    elif wants_voice or pres.get("speaks"):
        reasons: list[str] = []
        if motor_will:
            reasons.append("motore pronto")
        if pres.get("speaks"):
            reasons.append("vuole parlare")
        if pressure > 0.12:
            reasons.append("pensiero forte")
        if is_question(heard or ""):
            reasons.append("domanda ricevuta")
        lines.append(f"tace ma dentro: {', '.join(reasons) or 'impulso'}")
    else:
        lines.append("tace · osserva in silenzio")

    return lines


_NOISE_LINES = frozenset(
    {
        "tace · osserva in silenzio",
    }
)


def is_consciousness_noise(line: str) -> bool:
    s = line.strip()
    if not s or s == "[pulse]":
        return True
    if s in _NOISE_LINES:
        return True
    if s.startswith("pre-coscienza (0.00)") and "modo flow" in s:
        return True
    return False


def parse_consciousness_line(line: str, *, seq: int = 0) -> dict[str, Any]:
    """Trasforma una riga grezza in evento leggibile per la UI."""
    raw = line.strip()
    if not raw:
        return {"seq": seq, "kind": "skip", "raw": raw}

    if raw == "[pulse]" or raw.startswith("[") and raw.endswith("]"):
        phase = raw.strip("[]")
        return {"seq": seq, "kind": "phase", "icon": "⚡", "title": phase, "raw": raw}

    if raw.startswith("udito"):
        detail = _extract_quoted(raw) or raw.split(":", 1)[-1].strip()
        q = "domanda" in raw.lower()
        return {
            "seq": seq,
            "kind": "heard",
            "icon": "👂",
            "title": "Ha sentito" + (" (domanda)" if q else ""),
            "detail": detail,
            "raw": raw,
        }

    if raw.startswith("pensiero spontaneo:"):
        detail = raw.split(":", 1)[-1].strip()
        return {"seq": seq, "kind": "spontaneous", "icon": "💭", "title": "Pensiero spontaneo", "detail": detail, "raw": raw}

    if raw.startswith("pensa:") or raw.startswith("pensiero:"):
        detail = raw.split(":", 1)[-1].strip()
        return {"seq": seq, "kind": "think", "icon": "🧠", "title": "Pensa", "detail": detail, "raw": raw}

    if raw.startswith("esprime"):
        detail = _extract_quoted(raw) or raw.split(":", 1)[-1].strip()
        return {"seq": seq, "kind": "speak", "icon": "🗣️", "title": "Parla", "detail": detail, "raw": raw}

    if raw.startswith("sé:") or raw.startswith("sé "):
        detail = raw.split(":", 1)[-1].strip() if ":" in raw else raw[2:].strip()
        return {"seq": seq, "kind": "self", "icon": "🪞", "title": "Sé", "detail": detail, "raw": raw}

    if raw.startswith("coscienza") or raw.startswith("pre-coscienza"):
        return {"seq": seq, "kind": "awareness", "icon": "✨", "title": "Coscienza", "detail": raw, "raw": raw}

    if raw.startswith("sentimento:"):
        detail = raw.split(":", 1)[-1].strip()
        return {"seq": seq, "kind": "emotion", "icon": "❤️", "title": "Sentimento", "detail": detail, "raw": raw}

    if raw.startswith("onda:"):
        detail = raw.split(":", 1)[-1].strip()
        return {"seq": seq, "kind": "wave", "icon": "〰️", "title": "Onda cerebrale", "detail": detail, "raw": raw}

    if raw.startswith("tace"):
        return {"seq": seq, "kind": "silent", "icon": "🤫", "title": "Silenzio", "detail": "osserva", "raw": raw}

    if raw.startswith("  ↳") or raw.startswith("  ◉"):
        detail = raw.strip()
        return {"seq": seq, "kind": "assoc", "icon": "↳", "title": "Associazione", "detail": detail, "raw": raw}

    return {"seq": seq, "kind": "note", "icon": "·", "title": "Nota", "detail": raw, "raw": raw}


def _extract_quoted(text: str) -> str:
    m = re.search(r"«([^»]*)»", text)
    if m:
        return m.group(1).strip()
    m = re.search(r'"([^"]*)"', text)
    return m.group(1).strip() if m else ""


def consciousness_events_from_log(
    lines: list[str],
    *,
    since_seq: int = 0,
    limit: int = 48,
    include_noise: bool = False,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    seq = 0
    for line in lines:
        seq += 1
        if seq <= since_seq:
            continue
        if not include_noise and is_consciousness_noise(line):
            continue
        ev = parse_consciousness_line(line, seq=seq)
        if ev.get("kind") == "skip":
            continue
        ev["t"] = time.time()
        events.append(ev)
    tail = events[-limit:]
    return {
        "seq": seq,
        "events": tail,
        "has_more": len(events) > limit,
    }
