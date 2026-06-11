"""Presenza umana — vergogna, orario, voglia di parlare."""

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from organism.drives.presence import HumanPresence, _schedule_openness

TZ = ZoneInfo("Europe/Madrid")


def test_night_low_openness():
    assert _schedule_openness(2) < 0.3
    assert _schedule_openness(14) > 0.7


def test_shame_blocks_speech_in_new_scene():
    p = HumanPresence()
    s = p.evaluate(
        curiosity=0.9,
        novelty=1.0,
        boredom=0.8,
        stimulus_key="new_scene",
        visual_energy=0.6,
        impulse="vocalize",
    )
    assert s.shame > 0.4
    assert s.speaks is False or s.urge < 0.7


def test_familiar_scene_more_open():
    p = HumanPresence()
    p._familiar_scenes.add("home")
    s = p.evaluate(
        curiosity=0.7,
        novelty=0.1,
        boredom=0.5,
        stimulus_key="home",
        visual_energy=0.2,
        impulse="vocalize",
    )
    assert s.comfort > 0.5


@patch("organism.drives.presence.datetime")
def test_schedule_affects_mood(mock_dt):
    mock_dt.now.return_value = datetime(2026, 6, 7, 3, 0, tzinfo=TZ)
    p = HumanPresence()
    s = p.evaluate(
        curiosity=0.5,
        novelty=0.2,
        boredom=0.3,
        stimulus_key="x",
        impulse="attend",
    )
    assert s.openness < 0.25
    assert s.mood == "sleeping"
