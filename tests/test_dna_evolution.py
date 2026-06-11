from organism.dna.evolution import DNAEvolver


def test_evolver_scores_probes():
    ev = DNAEvolver()
    metrics = {
        "probe_outputs": [
            ("ciao", "Vedo il mondo e noto che sto imparando ogni giorno."),
            ("chi sei", "Sono un organismo che pensa e osserva con attenzione."),
        ],
        "speech_diversity": 1.0,
        "thought_coherence": 0.6,
    }
    r = ev.evaluate_metrics(metrics)
    assert r["score"] > 0.4
    assert r["avg_words"] >= 6


def test_export_clean_no_learned():
    ev = DNAEvolver()
    clean = ev.export_clean()
    assert clean.get("pattern_lexicon") == {}
    assert clean.get("baby", {}).get("empty_knowledge") is True
