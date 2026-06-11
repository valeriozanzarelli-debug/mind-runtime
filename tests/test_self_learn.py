from organism.cognition.self_learn import SelfLearner
from organism.motor.compose_speech import SpeechComposer
from organism.cognition.thought import Thought


def test_detect_unknown():
    sl = SelfLearner()
    uw = sl.detect_unknown("cos'è la fotosintesi", has_pathway=False)
    assert "fotosintesi" in uw or len(uw) > 0


def test_curiosity_question():
    from organism.motor.emergent_speech import EmergentSpeechMotor
    from organism.runtime import OrganismRuntime

    c = SpeechComposer(seed=1)
    org = OrganismRuntime.baby(seed=1)
    motor = EmergentSpeechMotor(org.brain, seed=1)
    c.bind(org.brain, motor)
    for w in ("non", "so", "cosa", "significa", "fotosintesi"):
        c.absorb(w, boost=2.0)
    t = Thought(
        symbols=["IMPULSE:ask", "UNKNOWN:fotosintesi"],
        themes=["non", "so", "fotosintesi"],
        pressure=0.6,
    )
    out = c._from_curiosity_question(t, motor)
    assert "?" in out
    assert "fotosintesi" in out.lower()
