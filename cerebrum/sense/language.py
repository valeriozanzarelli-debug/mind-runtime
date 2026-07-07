"""Udito/linguaggio in ingresso — trasforma il testo/parlato in stimolo.

Il neonato non capisce le parole: le vive come suoni con novità e intensità,
e lentamente associa token ricorrenti a stati interni (base per apprendere il
linguaggio). Manteniamo un piccolo lessico esperienziale.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict


class LanguageSense:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.heard_counts = defaultdict(int)
        self.token_affect = defaultdict(float)  # associazione token -> valenza

    def _tokenize(self, text: str):
        return [t for t in text.lower().replace("?", " ? ").replace("!", " ! ").split() if t]

    def encode(self, text: str) -> dict:
        tokens = self._tokenize(text)
        if not tokens:
            return {"intensity": 0.0, "novelty": 0.0, "vector": [0.0] * self.dim, "tokens": []}
        vec = [0.0] * self.dim
        novelty_sum = 0.0
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
            seen = self.heard_counts[tok]
            novelty_sum += 1.0 / (1.0 + seen)
            self.heard_counts[tok] += 1
        norm = max(sum(abs(v) for v in vec), 1e-6)
        vec = [v / norm for v in vec]
        intensity = min(len(tokens) / 12.0, 1.0)
        novelty = novelty_sum / len(tokens)
        return {"intensity": intensity, "novelty": novelty, "vector": vec, "tokens": tokens}

    def reinforce(self, tokens, valence: float) -> None:
        for tok in tokens:
            self.token_affect[tok] = 0.9 * self.token_affect[tok] + 0.1 * valence

    def vocabulary_size(self) -> int:
        return len(self.heard_counts)

    def known_tokens(self, n: int = 12):
        top = sorted(self.heard_counts.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [t for t, _ in top]
