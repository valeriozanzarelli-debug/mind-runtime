"""Curriculum programmazione — da snippet a programmi interi."""

from __future__ import annotations

import time
from typing import Any, Callable

# Coppie trigger → codice completo (Python didattico)
CODE_LESSONS: list[tuple[str, str, str]] = [
    (
        "stampa ciao",
        "print('ciao')",
        "programma che saluta",
    ),
    (
        "somma due numeri",
        "def somma(a, b):\n    return a + b\n\nprint(somma(3, 5))",
        "funzione somma",
    ),
    (
        "lista numeri",
        "numeri = [1, 2, 3, 4, 5]\nfor n in numeri:\n    print(n)",
        "ciclo su lista",
    ),
    (
        "conta fino a dieci",
        "for i in range(1, 11):\n    print(i)",
        "ciclo for",
    ),
    (
        "funzione quadrato",
        "def quadrato(x):\n    return x * x\n\nprint(quadrato(7))",
        "funzione quadrato",
    ),
    (
        "leggi input",
        "nome = input('come ti chiami? ')\nprint('ciao', nome)",
        "input utente",
    ),
    (
        "condizione maggiore",
        "def maggiore(a, b):\n    if a > b:\n        return a\n    return b",
        "if else",
    ),
    (
        "classe persona",
        "class Persona:\n    def __init__(self, nome):\n        self.nome = nome\n\n    def saluta(self):\n        return f'ciao sono {self.nome}'",
        "classe oop",
    ),
    (
        "file hello",
        "with open('hello.txt', 'w') as f:\n    f.write('ciao mondo')",
        "scrittura file",
    ),
    (
        "programma completo calcolatrice",
        "def calcola(a, op, b):\n    if op == '+':\n        return a + b\n    if op == '-':\n        return a - b\n    if op == '*':\n        return a * b\n    if op == '/':\n        return a / b if b else None\n\nprint(calcola(10, '+', 5))",
        "calcolatrice completa",
    ),
]


def run_code_curriculum(
    teach_fn: Callable[..., dict[str, Any]],
    *,
    pause_s: float = 0.3,
) -> dict[str, Any]:
    """teach_fn(when, say, kind='code')"""
    log: list[dict[str, Any]] = []
    taught = 0
    for trigger, code, label in CODE_LESSONS:
        try:
            r = teach_fn(trigger, code, kind="code")
            log.append({"trigger": trigger, "label": label, **r})
            if r.get("learned") or r.get("consolidated") or int(r.get("dialogue_pairs", 0)) > 0:
                taught += 1
            elif r.get("count", 0):
                taught += 1
        except Exception as e:
            log.append({"trigger": trigger, "error": str(e)[:80]})
        time.sleep(pause_s)
    return {"ok": taught > 0, "taught": taught, "lessons": log}
