from organism.cognition.comprehension import ComprehensionFrame
from organism.cognition.superego import SuperegoEngine


def test_superego_blocks_social_lexicon_dump():
    sg = SuperegoEngine()
    frame = ComprehensionFrame(
        intent="social",
        depth=0.35,
        confidence=0.9,
        inhibit_lexicon_dump=True,
    )
    bad = "So legno corvo formaggio volpe ciliegia lusinga catasta bottega becco."
    v = sg.review(bad, frame=frame)
    assert v.action == "substitute"
    assert "ascolto" in v.text.lower()


def test_superego_blocks_story_on_vision():
    sg = SuperegoEngine()
    frame = ComprehensionFrame(
        intent="vision",
        depth=0.5,
        confidence=0.6,
        inhibit_lexicon_dump=True,
    )
    bad = "pinocchio formaggio legno?"
    v = sg.review(bad, heard="", frame=frame)
    assert v.action == "block"
    assert "pinocchio" not in v.text.lower()


def test_superego_blocks_pinocchio_on_smalltalk():
    sg = SuperegoEngine()
    frame = ComprehensionFrame(
        intent="social",
        depth=0.4,
        confidence=0.85,
        heard="come stai",
        taught_say="sto bene, grazie.",
        inhibit_lexicon_dump=True,
    )
    bad = "Pinocchio legno poi bottega geppetto fra catasta corvo."
    v = sg.review(bad, heard="come stai", frame=frame)
    assert v.action in ("block", "substitute")
    assert "pinocchio" not in v.text.lower() or "sicuro" in v.text.lower()


def test_superego_internalizes_correction():
    sg = SuperegoEngine()
    sg.internalize("pinocchio è legno", "pinocchio è un burattino di legno")
    v = sg.review("pinocchio è legno materiale", frame=None)
    assert v.action == "substitute"
    assert "burattino" in v.text
