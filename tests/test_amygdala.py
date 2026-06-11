"""Amigdala — curiosità, gioco, paura modulano impulsi."""

from organism.cognition.amygdala import AmygdalaEngine
from organism.drives.curiosity import CuriosityDrive, CuriosityState
from organism.runtime import OrganismRuntime


def test_amygdala_boosts_explore_on_uncertainty():
    org = OrganismRuntime.baby(seed=1)
    amy = AmygdalaEngine()
    cur = CuriosityDrive()
    cur.state = CuriosityState(novelty=0.8, uncertainty=0.75, boredom=0.1)
    amy.pulse(
        org.brain,
        joy=0.5,
        fear=0.1,
        shame=0.05,
        anger=0.0,
        trust=0.5,
        curiosity=0.7,
        curiosity_drive=cur.state,
    )
    assert amy.state.explore > 0.45
    impulse = cur.choose_impulse(amygdala=amy)
    assert impulse in ("ask", "investigate", "vocalize")


def test_amygdala_fear_suppresses_vocalize():
    amy = AmygdalaEngine()
    amy.state.avoid = 0.75
    cur = CuriosityDrive()
    cur.state.boredom = 0.8
    assert cur.choose_impulse(amygdala=amy) == "attend"
