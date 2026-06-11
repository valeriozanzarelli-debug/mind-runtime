from organism.cognition.theme_coherence import coherent_themes


def test_filters_noise():
    themes = ["ciao", "domani", "loop", "come", "stai", "python"]
    out = coherent_themes(themes, heard="ciao come stai", pathway_words=["sto", "bene", "grazie"])
    assert "ciao" in out or "come" in out
    assert "python" not in out
    assert "domani" not in out


def test_pathway_keeps_relevant():
    themes = ["python", "domani", "bene", "grazie", "ciao"]
    out = coherent_themes(
        themes,
        heard="grazie",
        pathway_words=["prego", "niente"],
        min_score=0.5,
    )
    assert "grazie" in out
    assert "python" not in out
