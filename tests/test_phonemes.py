"""Voce emergente — sillabe da udito, attivazione neurale."""

from organism.motor.emergent_speech import EmergentSpeechMotor
from organism.teaching.phonemes import PhonemeLearner, split_italian_syllables


def test_syllables_from_phrase():
    syl = split_italian_syllables("ciao come stai")
    assert syl


def test_phoneme_learner_empty_until_hear():
    p = PhonemeLearner()
    assert p.count == 0
    p.hear("ciao")
    assert p.count > 0


def test_emergent_speech_after_hearing():
    from organism.runtime import OrganismRuntime

    org = OrganismRuntime.baby(seed=1)
    motor = EmergentSpeechMotor(org.brain, seed=1)
    assert motor.utter() == ""
    motor.hear("mamma")
    for n in motor.phoneme_neurons[:4]:
        n.activation = 0.65
    for n in org.brain.get_neurons("motor", "speech_phoneme_generator"):
        n.activation = max(n.activation, 0.4)
    spoke = motor.utter()
    assert spoke
    assert "lampadina" not in spoke.lower()
