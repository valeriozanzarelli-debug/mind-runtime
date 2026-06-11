"""Flusso di coscienza e domande WH."""

from organism.autonomous.baby_agent import BabyAgent
from organism.cognition.consciousness_stream import build_consciousness_stream, is_question


def test_is_question():
    assert is_question("chi sei?")
    assert is_question("come stai")
    assert not is_question("ciao")


def test_stream_when_silent():
    lines = build_consciousness_stream(
        heard="chi sei",
        thought={"themes": ["chi", "organism"], "symbols": ["QUESTION:yes"], "pressure": 0.4},
        workspace={"conscious": True, "ignition": 0.35, "mode": "speak", "focus": "chi"},
        spoke="",
        wants_voice=True,
        presence={"speaks": True},
        motor_will=True,
    )
    text = "\n".join(lines)
    assert "udito" in text
    assert "pensa:" in text
    assert "tace ma dentro" in text or "domanda" in text


def test_wh_question_gets_response(tmp_path):
    b = BabyAgent(seed=9, store_path=str(tmp_path / "q.json"))
    b.birth()
    for _ in range(3):
        b.teach_dialogue("chi sei", "sono organism")
    m = b.sense(text="chi sei")["moment"]
    assert m["consciousness_stream"]
    assert m["spoke"] or m["wanted_to_speak"] or m["understood"]
