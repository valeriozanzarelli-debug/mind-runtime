#!/usr/bin/env python3
"""Sessione Hebbiana — chat + sleep tra ogni turno + benchmark periodici.

NON usa teach-dialogue. Solo chat, absorb, sleep, fluency-benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BASE = "https://inkconscius.eu/organism"
LOG_DIR = Path("/tmp/baby_hebbian_sessions")


@dataclass
class SessionMetrics:
    turn: int = 0
    benchmarks: list[dict] = field(default_factory=list)
    synapses_start: int = 0
    synapses_end: int = 0
    words_start: int = 0
    words_end: int = 0


def post(base: str, path: str, body: dict | None = None, *, timeout: float = 180) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_state(base: str) -> dict:
    with urllib.request.urlopen(f"{base.rstrip('/')}/api/baby/state", timeout=60) as resp:
        return json.loads(resp.read().decode())


def snapshot_state(base: str) -> tuple[int, int]:
    st = get_state(base)
    syn = int(st.get("stats", {}).get("synapses") or st.get("brain", {}).get("synapses") or 0)
    words = int(st.get("words_known") or 0)
    return syn, words


def run_benchmark(base: str, *, limit: int = 30) -> dict:
    try:
        return post(
            base,
            "/api/baby/fluency-benchmark",
            {"limit": limit, "pause_s": 0.08},
            timeout=600,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def grammar_probe(base: str) -> dict:
    """Probe rapidi su grammatica e comprensione."""
    probes = [
        ("ciao", "saluto"),
        ("il gatto dorme sul divano", "comprensione_svo"),
        ("cos'è l'acqua", "concetto"),
        ("perché piove", "causa_effetto"),
        ("quali colori conosci", "vocabolario"),
        ("raccontami qualcosa", "generazione"),
        ("come ti senti", "emozione"),
        ("cosa vedi intorno a te", "percezione"),
    ]
    results = []
    coherent = svo = relevant = 0
    for prompt, kind in probes:
        try:
            r = post(base, "/api/baby/chat", {"text": prompt}, timeout=120)
            reply = str(r.get("reply") or r.get("moment", {}).get("spoke") or "")
            words = len(reply.split())
            has_svo = bool(
                any(v in reply.lower() for v in ("sono", "è", "ho", "vedo", "penso", "sento", "credo"))
                and words >= 4
            )
            not_babble = words >= 3 and not reply.lower().startswith("acqua arancione")
            if has_svo:
                svo += 1
            if not_babble and words >= 5:
                coherent += 1
            if kind == "saluto" and any(w in reply.lower() for w in ("ciao", "qui", "ciao!")):
                relevant += 1
            elif kind != "saluto" and not_babble:
                relevant += 1
            results.append({"prompt": prompt, "kind": kind, "reply": reply[:140], "words": words, "svo": has_svo})
        except Exception as exc:
            results.append({"prompt": prompt, "kind": kind, "error": str(exc)[:80]})
        time.sleep(0.5)
    n = max(1, len(probes))
    return {
        "coherent_ratio": round(coherent / n, 3),
        "svo_ratio": round(svo / n, 3),
        "relevant_ratio": round(relevant / n, 3),
        "probes": results,
    }


def log_line(log_path: Path, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def training_messages() -> list[str]:
    return [
        "Ciao Baby, sono il tuo educatore. Parliamo un po'.",
        "Il sole scalda la terra durante il giorno.",
        "I gatti amano dormire al caldo.",
        "L'acqua è trasparente e la beviamo quando abbiamo sete.",
        "Quando piove le nuvole sono grigie.",
        "Gli alberi hanno foglie verdi in estate.",
        "Il rosso è il colore delle rose.",
        "La casa è dove ci riposiamo la sera.",
        "Sono contento di insegnarti cose nuove.",
        "Ogni parola che senti ti aiuta a capire.",
        "Il mare è grande e ha l'acqua salata.",
        "Un fiore sboccia in primavera.",
        "La luna illumina la notte.",
        "Il pane profuma buono quando è fresco.",
        "Mi incuriosisce come impari piano piano.",
        "Dimmi cosa pensi in questo momento.",
        "Il vento muove le foglie degli alberi.",
        "In montagna l'aria è fresca e pulita.",
        "Un cane abbaia quando vede qualcuno.",
        "La pioggia bagna la strada.",
        "Il fuoco dà calore ma va usato con attenzione.",
        "Leggere un libro apre la mente.",
        "La musica fa vibrare le emozioni.",
        "Di notte il cielo ha molte stelle.",
        "Imparare richiede tempo e pazienza.",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Hebbian training session with sleep between turns")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--turns", type=int, default=20, help="Numero turni chat")
    parser.add_argument("--benchmark-every", type=int, default=5, help="Benchmark ogni N turni")
    parser.add_argument("--absorb-every", type=int, default=8, help="Absorb Wikipedia ogni N turni")
    parser.add_argument("--sleep-after-each", action="store_true", default=True)
    parser.add_argument("--session-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    base = args.base.rstrip("/")
    log_path = LOG_DIR / f"session_{args.session_id}.log"
    metrics = SessionMetrics()

    ready = urllib.request.urlopen(f"{base}/api/baby/ready", timeout=30)
    if ready.status != 200:
        print("Baby non pronto", file=sys.stderr)
        return 1

    metrics.synapses_start, metrics.words_start = snapshot_state(base)
    log_line(log_path, f"=== SESSIONE {args.session_id} ===")
    log_line(log_path, f"sinapsi={metrics.synapses_start} parole={metrics.words_start}")

    initial_bm = run_benchmark(base, limit=20)
    initial_gp = grammar_probe(base)
    metrics.benchmarks.append({
        "turn": 0,
        "fluency": initial_bm.get("aggregate", initial_bm),
        "grammar": initial_gp,
    })
    log_line(
        log_path,
        f"BENCHMARK iniziale: coherent={initial_gp['coherent_ratio']:.0%} "
        f"svo={initial_gp['svo_ratio']:.0%} relevant={initial_gp['relevant_ratio']:.0%}",
    )

    messages = training_messages()
    topics_cycle = ["gatto", "acqua", "pioggia", "colore", "stella", "fiore", "casa", "emozione"]

    for i in range(args.turns):
        metrics.turn = i + 1
        msg = messages[i % len(messages)]

        try:
            r = post(base, "/api/baby/chat", {"text": msg}, timeout=120)
            reply = str(r.get("reply") or "")
            log_line(log_path, f"T{i+1}: {msg[:60]}")
            log_line(log_path, f"B{i+1}: {reply[:120]}")
        except Exception as exc:
            log_line(log_path, f"ERR turno {i+1}: {exc}")
            time.sleep(5)
            continue

        if args.sleep_after_each:
            try:
                sl = post(base, "/api/baby/sleep", timeout=60)
                pruned = sl.get("pruned_synapses", 0)
                syn_after = sl.get("synapses_after", "?")
                log_line(log_path, f"SLEEP: potate={pruned} sinapsi={syn_after}")
            except Exception as exc:
                log_line(log_path, f"SLEEP err: {exc}")
            time.sleep(1)

        if args.absorb_every and (i + 1) % args.absorb_every == 0:
            topic = topics_cycle[(i // args.absorb_every) % len(topics_cycle)]
            try:
                tw = post(base, "/api/baby/train-wikipedia", {"topics": [topic]}, timeout=180)
                log_line(log_path, f"WIKI [{topic}]: {tw.get('absorbed_chars', 0)} chars")
            except Exception as exc:
                log_line(log_path, f"WIKI err: {exc}")

        if args.benchmark_every and (i + 1) % args.benchmark_every == 0:
            gp = grammar_probe(base)
            bm = run_benchmark(base, limit=15)
            entry = {
                "turn": i + 1,
                "grammar": gp,
                "fluency": bm.get("aggregate", bm),
            }
            metrics.benchmarks.append(entry)
            agg = bm.get("aggregate") or {}
            log_line(
                log_path,
                f"BENCHMARK @{i+1}: grammar_coherent={gp['coherent_ratio']:.0%} "
                f"svo={gp['svo_ratio']:.0%} "
                f"fluency_mean={agg.get('mean_score', agg.get('score_mean', '?'))}",
            )

    metrics.synapses_end, metrics.words_end = snapshot_state(base)
    final_gp = grammar_probe(base)
    final_bm = run_benchmark(base, limit=25)

    log_line(log_path, "=== FINE SESSIONE ===")
    log_line(
        log_path,
        f"sinapsi {metrics.synapses_start} → {metrics.synapses_end} "
        f"| parole {metrics.words_start} → {metrics.words_end}",
    )
    log_line(
        log_path,
        f"FINALE grammar: coherent={final_gp['coherent_ratio']:.0%} "
        f"svo={final_gp['svo_ratio']:.0%} relevant={final_gp['relevant_ratio']:.0%}",
    )

    report_path = LOG_DIR / f"report_{args.session_id}.json"
    report = {
        "session_id": args.session_id,
        "turns": metrics.turn,
        "synapses": {"start": metrics.synapses_start, "end": metrics.synapses_end},
        "words": {"start": metrics.words_start, "end": metrics.words_end},
        "benchmarks": metrics.benchmarks,
        "final_grammar": final_gp,
        "final_fluency": final_bm,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log_line(log_path, f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
