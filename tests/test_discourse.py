from organism.cognition.discourse import compose_discourse, is_babble


def test_compose_discourse_min_length():
    text = compose_discourse(["mela", "albero", "mare", "cielo"], min_words=8)
    assert len(text.split()) >= 8
    assert text.endswith(".")


def test_is_babble_rejects_short():
    assert is_babble("ciao ciao")
    assert not is_babble("vedo una mela rossa sull albero e capisco")


def test_is_babble_rejects_repeat():
    assert is_babble("parola parola parola parola parola")
