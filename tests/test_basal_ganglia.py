from organism.cognition.basal_ganglia import BasalGanglia


def test_basal_ganglia_prefers_speak_when_heard():
    bg = BasalGanglia()
    sel = bg.select(
        default="silent",
        heard="ciao",
        curiosity=0.4,
        novelty=0.2,
        boredom=0.1,
        amygdala_inhibition=0.1,
        has_association=True,
        wants_ask=False,
    )
    assert sel.impulse in ("speak", "ask", "vocalize")
