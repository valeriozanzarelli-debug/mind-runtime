from organism.cognition.human_thought import compose_inner_voice, thought_coherence


def test_inner_voice_first_person():
    t = compose_inner_voice(themes=["mare", "cielo", "luce"], heard="cosa vedi")
    assert " " in t
    assert t.endswith(".")
    assert len(t.split()) >= 6


def test_thought_coherence_with_heard():
    c = thought_coherence(["mare", "cielo"], heard="vedi il mare")
    assert c > 0.3
