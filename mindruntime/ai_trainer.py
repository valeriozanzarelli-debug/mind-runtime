"""Canale AI training — stub export/import per apprendimento supervisionato opzionale."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False


@dataclass
class TrainStepResult:
    loss: float | None
    predicted: str | None
    weight_delta: np.ndarray | None
    message: str = ""


if HAS_TORCH:

    class _TinyHead(nn.Module):  # type: ignore[misc]
        def __init__(self, n_classes: int, spatial: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, n_classes),
            )
            self._spatial = spatial

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            return self.net(x)

else:

    class _TinyHead:  # type: ignore[no-redef]
        pass


class AITrainer:
    """Interfaccia training — funziona senza PyTorch (solo export/import)."""

    def __init__(self, class_names: list[str] | None = None) -> None:
        self.class_names = class_names or []
        self._model: Any = None
        self._optim: Any = None
        if HAS_TORCH and self.class_names:
            self._model = _TinyHead(len(self.class_names), 32)
            self._optim = torch.optim.Adam(self._model.parameters(), lr=1e-3)

    def state_to_tensor(self, state: dict[str, np.ndarray]) -> Any:
        if not HAS_TORCH:
            return None
        planes = np.stack(
            [
                state["impulse"],
                np.sin(state["phase"]),
                np.cos(state["phase"]),
                state["weight"],
            ],
            axis=0,
        )
        t = torch.from_numpy(planes.astype(np.float32)).unsqueeze(0)
        return t

    def train_step(
        self,
        state: dict[str, np.ndarray],
        label: str | int | None = None,
    ) -> TrainStepResult:
        """Un passo supervisionato opzionale → gradiente sui pesi GPU."""
        if not HAS_TORCH or self._model is None or label is None:
            return TrainStepResult(
                loss=None,
                predicted=None,
                weight_delta=None,
                message="training supervisionato disabilitato (installa torch + class_names)",
            )
        if isinstance(label, str):
            if label not in self.class_names:
                return TrainStepResult(None, None, None, f"label sconosciuta: {label}")
            target = self.class_names.index(label)
        else:
            target = int(label)

        x = self.state_to_tensor(state)
        assert x is not None
        y = torch.tensor([target], dtype=torch.long)
        logits = self._model(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        # Mappa gradiente input → delta pesi (semplificato)
        if x.grad is not None:
            delta = x.grad[0, 3].numpy() * 0.01
        else:
            x.requires_grad_(True)
            logits = self._model(x)
            loss = loss_fn(logits, y)
            self._optim.zero_grad()
            loss.backward()
            delta = (x.grad[0, 3].numpy() * 0.01) if x.grad is not None else np.zeros_like(state["weight"])

        pred_idx = int(logits.argmax(dim=1).item())
        return TrainStepResult(
            loss=float(loss.item()),
            predicted=self.class_names[pred_idx],
            weight_delta=delta.astype(np.float32),
            message="ok",
        )

    def update_resonators(self, engine: Any, new_templates: np.ndarray, names: list[str]) -> None:
        engine.update_resonators(new_templates, names)
