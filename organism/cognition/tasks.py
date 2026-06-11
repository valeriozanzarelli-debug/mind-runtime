"""Compiti — responsabilità, portare a termine, imparare dagli errori."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

TaskKind = Literal["repeat", "name_scene", "read_aloud", "answer"]
TaskStatus = Literal["pending", "active", "done", "failed"]


@dataclass
class Task:
    id: str
    kind: TaskKind
    prompt: str
    target: str
    status: TaskStatus = "pending"
    attempts: int = 0
    last_spoke: str = ""
    score: float = 0.0
    created_t: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "target": self.target,
            "status": self.status,
            "attempts": self.attempts,
            "last_spoke": self.last_spoke[:120],
            "score": round(self.score, 3),
        }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w{2,}", text.lower()))


class TaskRunner:
    """Assegna compiti; valuta solo overlap su ciò che è stato insegnato."""

    def __init__(self) -> None:
        self._tasks: list[Task] = []
        self._active_id: str | None = None

    def assign(self, kind: TaskKind, prompt: str, target: str) -> Task:
        t = Task(
            id=uuid.uuid4().hex[:10],
            kind=kind,
            prompt=prompt.strip(),
            target=target.strip(),
            status="active",
        )
        self._tasks.append(t)
        self._active_id = t.id
        return t

    def active(self) -> Task | None:
        if not self._active_id:
            return None
        for t in self._tasks:
            if t.id == self._active_id and t.status == "active":
                return t
        return None

    def evaluate_attempt(self, spoke: str, *, corrected: bool = False) -> dict[str, Any]:
        t = self.active()
        if not t:
            return {"active": False}
        t.attempts += 1
        t.last_spoke = spoke
        exp = _tokens(t.target)
        got = _tokens(spoke)
        overlap = len(exp & got) / max(1, len(exp)) if exp else 0.0
        t.score = overlap
        if corrected:
            t.score = min(1.0, overlap + 0.35)
        if overlap >= 0.45 or (corrected and overlap >= 0.25):
            t.status = "done"
            self._active_id = None
            return {"active": True, "done": True, "score": t.score, "task": t.to_dict()}
        if t.attempts >= 6:
            t.status = "failed"
            self._active_id = None
            return {"active": True, "done": False, "failed": True, "score": t.score, "task": t.to_dict()}
        return {"active": True, "done": False, "score": t.score, "task": t.to_dict()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self._active_id,
            "tasks": [t.to_dict() for t in self._tasks[-20:]],
            "done": sum(1 for t in self._tasks if t.status == "done"),
            "failed": sum(1 for t in self._tasks if t.status == "failed"),
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._active_id = data.get("active")
        self._tasks = []
        for d in data.get("tasks", []):
            self._tasks.append(
                Task(
                    id=str(d.get("id", "")),
                    kind=d.get("kind", "repeat"),  # type: ignore[arg-type]
                    prompt=str(d.get("prompt", "")),
                    target=str(d.get("target", "")),
                    status=d.get("status", "done"),  # type: ignore[arg-type]
                    attempts=int(d.get("attempts", 0)),
                    last_spoke=str(d.get("last_spoke", "")),
                    score=float(d.get("score", 0)),
                )
            )
