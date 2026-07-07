"""Predictive coding — il motore del pensiero.

Un cervello passa il tempo a PREDIRE il prossimo istante sensoriale e a
reagire solo all'ERRORE di predizione. Questo genera:
- curiosita' (errore alto = qualcosa di nuovo da imparare),
- attenzione (dove sbaglio di piu'),
- pensiero anche senza stimoli (a occhi chiusi il cervello continua a
  generare l'input che si aspetta -> immaginazione).

Implementazione minimale ma reale: un predittore lineare che, dall'attivita'
associativa, predice un riassunto sensoriale a dimensione fissa. Impara con la
regola delta (apprendimento locale, niente backprop globale). L'errore medio
e' la "sorpresa" che modula dopamina e curiosita'.
"""
from __future__ import annotations

import numpy as np


class PredictiveCoder:
    def __init__(self, assoc_dim: int = 64, sensory_dim: int = 64, seed: int = 3):
        rng = np.random.default_rng(seed)
        self.assoc_dim = assoc_dim
        self.sensory_dim = sensory_dim
        # matrice di predizione (sensory_dim x assoc_dim)
        self.W = (rng.standard_normal((sensory_dim, assoc_dim)) * 0.01).astype(np.float32)
        self.lr = 0.02
        self.surprise = 0.0
        self.mean_surprise = 0.1  # baseline mobile della sorpresa
        self._last_pred = np.zeros(sensory_dim, dtype=np.float32)

    def predict(self, assoc: np.ndarray) -> np.ndarray:
        a = np.asarray(assoc, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.assoc_dim:
            a = self._fit(a, self.assoc_dim)
        pred = self.W @ a
        self._last_pred = np.clip(pred, 0.0, 1.0)
        return self._last_pred

    def learn(self, assoc: np.ndarray, actual: np.ndarray) -> float:
        """Impara dall'errore di predizione. Ritorna la sorpresa (0..1)."""
        a = np.asarray(assoc, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.assoc_dim:
            a = self._fit(a, self.assoc_dim)
        target = np.asarray(actual, dtype=np.float32).reshape(-1)
        if target.shape[0] != self.sensory_dim:
            target = self._fit(target, self.sensory_dim)
        pred = self.W @ a
        err = target - pred
        # regola delta: aggiornamento locale
        self.W += self.lr * np.outer(err, a)
        np.clip(self.W, -4.0, 4.0, out=self.W)
        s = float(np.mean(np.abs(err)))
        self.surprise = s
        self.mean_surprise = 0.99 * self.mean_surprise + 0.01 * s
        return s

    def imagine(self, assoc: np.ndarray) -> np.ndarray:
        """A occhi chiusi: genera l'input atteso come 'pensiero' interno."""
        return self.predict(assoc)

    def relative_surprise(self) -> float:
        """Sorpresa normalizzata rispetto al baseline (novita' percepita)."""
        if self.mean_surprise <= 1e-6:
            return 0.0
        return float(np.clip(self.surprise / (self.mean_surprise + 1e-6) - 1.0, 0.0, 1.0))

    @staticmethod
    def _fit(v: np.ndarray, dim: int) -> np.ndarray:
        v = v.reshape(-1)
        if v.shape[0] == dim:
            return v
        if v.shape[0] == 0:
            return np.zeros(dim, dtype=np.float32)
        # ricampiona con media a blocchi / tiling
        if v.shape[0] > dim:
            return np.array([float(c.mean()) for c in np.array_split(v, dim)], dtype=np.float32)
        reps = int(np.ceil(dim / v.shape[0]))
        return np.tile(v, reps)[:dim].astype(np.float32)
