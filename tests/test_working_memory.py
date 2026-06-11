from organism.cognition.working_memory import WorkingMemory


def test_working_memory_slots():
    wm = WorkingMemory(capacity=3)
    wm.push(["mare", "blu"], heard="vedo il mare")
    wm.push(["albero", "verde"], heard="questo albero")
    ctx = wm.context_words()
    assert "mare" in ctx
    assert "albero" in ctx
    assert len(wm._slots) <= 3


def test_lexicon_squash():
    from organism.cognition.neural_lexicon import NeuralLexicon

    lex = NeuralLexicon()
    lex._exposure["ciao"] = 500.0
    n = lex.squash_overexposed()
    assert n == 1
    assert lex._exposure["ciao"] < 200.0
