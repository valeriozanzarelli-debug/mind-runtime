"""Visione — retina semplice che elabora i frame della webcam.

Accetta un frame come:
- lista/array RGB (H, W, 3) oppure grigio (H, W)
- oppure statistiche già estratte {brightness, motion, ...}

Estrae luminosità, movimento (differenza col frame precedente), contrasto e
un vettore 'retinico' a bassa dimensione che diventa corrente sensoriale per
il campo neurale. È la base per reagire agli stimoli visivi come un neonato.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class VisionSense:
    def __init__(self, patches: int = 64):
        self.patches = patches
        self._prev = None
        self.active = False
        self.last = {"brightness": 0.0, "motion": 0.0, "contrast": 0.0}

    def _to_gray(self, frame) -> Optional[np.ndarray]:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        elif arr.ndim == 1:
            side = int(np.sqrt(arr.shape[0]))
            if side * side == arr.shape[0]:
                arr = arr.reshape(side, side)
            else:
                return None
        elif arr.ndim != 2:
            return None
        m = arr.max()
        if m > 1.5:  # 0..255 -> 0..1
            arr = arr / 255.0
        return arr

    def process(self, frame=None, stats: Optional[dict] = None) -> dict:
        """Ritorna {brightness, motion, contrast, retina[list], active}."""
        if stats is not None:
            self.active = True
            b = float(stats.get("brightness", 0.0))
            mo = float(stats.get("motion", 0.0))
            c = float(stats.get("contrast", 0.0))
            retina = list(stats.get("retina", np.full(self.patches, b, dtype=np.float32)))
            self.last = {"brightness": b, "motion": mo, "contrast": c}
            return {"brightness": b, "motion": mo, "contrast": c,
                    "retina": retina[: self.patches], "active": True}

        gray = self._to_gray(frame) if frame is not None else None
        if gray is None:
            self.active = False
            return {"brightness": 0.0, "motion": 0.0, "contrast": 0.0,
                    "retina": [0.0] * self.patches, "active": False}

        self.active = True
        brightness = float(gray.mean())
        contrast = float(gray.std())
        if self._prev is not None and self._prev.shape == gray.shape:
            motion = float(np.abs(gray - self._prev).mean())
        else:
            motion = 0.0
        self._prev = gray

        # ridimensiona a griglia di patch per un vettore retinico stabile
        side = int(np.sqrt(self.patches))
        try:
            h, w = gray.shape
            hs, ws = max(h // side, 1), max(w // side, 1)
            small = gray[: hs * side, : ws * side].reshape(side, hs, side, ws).mean(axis=(1, 3))
            retina = small.reshape(-1)[: self.patches]
        except Exception:
            retina = np.full(self.patches, brightness, dtype=np.float32)
        if retina.shape[0] < self.patches:
            retina = np.pad(retina, (0, self.patches - retina.shape[0]))

        self.last = {"brightness": brightness, "motion": motion, "contrast": contrast}
        return {"brightness": brightness, "motion": motion, "contrast": contrast,
                "retina": [float(x) for x in retina], "active": True}
