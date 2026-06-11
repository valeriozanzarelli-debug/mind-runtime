from organism.cognition.syntax_planner import SyntaxPlanner
from organism.cognition.linguistic_narrator import LinguisticNarrator
from organism.cognition.thought import Thought


def test_syntax_planner_trains_bigrams():
    p = SyntaxPlanner()
    n = p.train_sentence("io voglio imparare a capire")
    assert n >= 4
    words = p.greedy_assembly(["imparare", "io", "voglio", "capire"])
    assert words[0] == "io"
    assert "voglio" in words or "imparare" in words


def test_narrator_produces_italian_sentence():
    narrator = LinguisticNarrator()
    narrator.bootstrap_curriculum(repeats=2)
    thought = Thought(
        themes=["io", "voglio", "imparare", "capire", "mondo"],
        pressure=0.7,
    )
    result = narrator.narrate(thought, force=True)
    assert result.from_layer2
    assert len(result.text.split()) >= 4
    assert result.text[0].isupper()


def test_curriculum_has_many_sentences():
    from organism.teaching.syntax_curriculum import curriculum_sentences

    sents = curriculum_sentences()
    assert len(sents) >= 100
