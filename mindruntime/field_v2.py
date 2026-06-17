"""Campo neurale V2 — 12 canali + codifica temporale tripla."""

from __future__ import annotations

import numpy as np

N_CH = 12
CH_IMP = 0      # rate coding
CH_PH = 1       # phase coding
CH_CA = 2
CH_NA = 3
CH_K = 4
CH_V = 5        # voltaggio mV
CH_W = 6        # peso gravità / SOC
CH_EN = 7       # energia attrattore
CH_COH = 8      # coherence / riconoscimento
CH_M = 9        # HH gating Na activation
CH_H = 10       # HH inactivation Na
CH_N = 11       # HH gating K

TWO_PI = 6.283185307179586
PI = 3.141592653589793


def field_zeros(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, N_CH), dtype=np.float32)


def spike_times_zeros(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.float32)
