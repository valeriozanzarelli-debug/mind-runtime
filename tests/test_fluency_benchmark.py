from organism.teaching.fluency_benchmark import (
    aggregate_scores,
    has_subject_verb,
    score_response,
    standard_prompts,
)


def test_standard_prompts_count():
    assert len(standard_prompts(limit=100)) == 100


def test_score_good_sentence():
    r = score_response(
        "cosa pensi",
        "io penso che sto imparando qualcosa di nuovo ogni giorno",
        themes=["imparare", "pensare", "nuovo"],
        from_thought=True,
    )
    assert not r.babble
    assert r.coherent
    assert r.words >= 8


def test_score_rejects_babble():
    r = score_response("ciao", "parola parola parola parola parola")
    assert r.babble
    assert not r.coherent


def test_has_subject_verb():
    assert has_subject_verb("io voglio imparare a capire")
    assert not has_subject_verb("rosso blu verde")


def test_aggregate():
    a = score_response("q", "io penso che il mare è bello e vasto oggi", from_thought=True)
    b = score_response("q2", "x")
    agg = aggregate_scores([a, b])
    assert agg["n"] == 2
    assert 0 < agg["coherent_ratio"] < 1
