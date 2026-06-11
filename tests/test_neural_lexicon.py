"""Lessico neurale — nessun lookup tabellare per parlare."""

from organism.cognition.neural_lexicon import NeuralLexicon
from organism.motor.compose_speech import SpeechComposer
from organism.cognition.thought import Thought
from organism.motor.emergent_speech import EmergentSpeechMotor
from organism.runtime import OrganismRuntime


def test_lexicon_articulable_after_exposure():
    org = OrganismRuntime.baby(seed=3)
    motor = EmergentSpeechMotor(org.brain, seed=3)
    lex = NeuralLexicon()
    lex.bind(org.brain, motor)
    lex.absorb("ciao come stai bene", boost=2.0)
    assert lex.is_articulable("ciao")
    assert lex.is_articulable("stai")


def test_compose_pathway_without_word_table():
    comp = SpeechComposer(seed=2)
    org = OrganismRuntime.baby(seed=2)
    motor = EmergentSpeechMotor(org.brain, seed=2)
    comp.bind(org.brain, motor)
    for w in ("ciao", "come", "stai", "bene", "grazie"):
        comp.absorb(w, boost=2.5)
    thought = Thought(themes=["ciao", "stai", "bene"], pressure=0.5, memory_hits=1)
    out = comp.produce(
        thought=thought,
        motor=motor,
        pathway_primed=True,
        pathway_words=["ciao", "come", "stai", "bene"],
    )
    assert out.text
    assert "ciao" in out.text.lower() or "stai" in out.text.lower()


def test_curiosity_question_neural():
    comp = SpeechComposer(seed=1)
    org = OrganismRuntime.baby(seed=1)
    motor = EmergentSpeechMotor(org.brain, seed=1)
    comp.bind(org.brain, motor)
    for w in ("non", "so", "cosa", "significa", "fotosintesi"):
        comp.absorb(w, boost=2.0)
    t = Thought(
        symbols=["IMPULSE:ask", "UNKNOWN:fotosintesi"],
        themes=["non", "so", "fotosintesi"],
        pressure=0.6,
    )
    out = comp._from_curiosity_question(t, motor)
    assert "?" in out
    assert "fotosintesi" in out.lower()
    assert "non so" not in out.lower() or "fotosintesi" in out.lower()


def test_lexical_readout_motor_assembly():
    org = OrganismRuntime.baby(seed=5)
    motor = EmergentSpeechMotor(org.brain, seed=5)
    for phrase in ("non so", "cosa significa", "spiega"):
        motor.hear(phrase, boost=2.0)
    plan = motor.lexical_readout(["non", "so", "galileo"], punct="?")
    assert plan.text.endswith("?")
    assert "galileo" in plan.text.lower()
    assert plan.from_motor
