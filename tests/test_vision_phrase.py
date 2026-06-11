from organism.teaching.vision_phrase import parse_vision_teaching


def test_parse_cassa_rosa():
    p = parse_vision_teaching("questa è una cassa rosa")
    assert p.object_name == "cassa"
    assert p.color in ("rosa", "rossa", "rosso")


def test_parse_mela():
    p = parse_vision_teaching("vedo una mela rossa")
    assert p.object_name == "mela"
    assert p.color in ("rosso", "rossa")


def test_parse_plain_phrase():
    p = parse_vision_teaching("ciao come stai")
    assert not p.has_object


def test_parse_quella_mela():
    p = parse_vision_teaching("quella è una mela")
    assert p.object_name == "mela"


def test_parse_e_una_cassa():
    p = parse_vision_teaching("è una cassa")
    assert p.object_name == "cassa"
