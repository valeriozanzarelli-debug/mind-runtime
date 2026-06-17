"""Memoria su disco — episodi illimitati oltre la RAM del server."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def vault_path() -> Path:
    raw = os.environ.get("ORGANISM_DISK_VAULT", "").strip()
    if raw and raw not in ("0", "1", "true", "false", "yes", "no"):
        return Path(raw)
    server = Path("/opt/mind-runtime/data/vault")
    if server.parent.parent.exists():
        return server
    return Path.home() / ".organism" / "vault"


@dataclass
class DiskMemoryVault:
    """Append-only JSONL — memoria a lungo termine su spazio disco del server."""

    root: Path = field(default_factory=vault_path)
    max_recall_scan: int = 4000

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._episodes = self.root / "episodes.jsonl"
        self._facts = self.root / "facts.jsonl"
        self._count = 0
        if self._episodes.exists():
            with self._episodes.open(encoding="utf-8") as f:
                for _ in f:
                    self._count += 1

    def append_episode(
        self,
        *,
        heard: str = "",
        spoke: str = "",
        themes: list[str] | None = None,
        emotion: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "t": time.time(),
            "heard": heard[:240],
            "spoke": spoke[:400],
            "themes": (themes or [])[:12],
            "emotion": emotion[:32],
            "meta": meta or {},
        }
        with self._episodes.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._count += 1

    def append_fact(self, key: str, value: str, *, source: str = "") -> None:
        row = {"t": time.time(), "key": key[:80], "value": value[:500], "source": source[:80]}
        with self._facts.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        if not query.strip() or not self._episodes.exists():
            return []
        tokens = {w for w in _tokens(query) if len(w) > 2}
        if not tokens:
            return []
        hits: list[tuple[float, dict[str, Any]]] = []
        scanned = 0
        with self._episodes.open(encoding="utf-8") as f:
            for line in f:
                scanned += 1
                if scanned > self.max_recall_scan:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = f"{row.get('heard','')} {row.get('spoke','')} {' '.join(row.get('themes',[]))}".lower()
                overlap = sum(1 for t in tokens if t in text)
                if overlap:
                    hits.append((overlap + overlap / max(1, len(tokens)), row))
        hits.sort(key=lambda x: -x[0])
        return [h[1] for h in hits[:limit]]

    def stats(self) -> dict[str, Any]:
        size = self._episodes.stat().st_size if self._episodes.exists() else 0
        facts = self._facts.stat().st_size if self._facts.exists() else 0
        return {
            "path": str(self.root),
            "episodes": self._count,
            "episodes_bytes": size,
            "facts_bytes": facts,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.stats()

    def load_dict(self, data: dict[str, Any]) -> None:
        self._count = int(data.get("episodes", self._count))


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-zàèéìòù']+", text.lower())
