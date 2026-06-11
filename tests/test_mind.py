"""Regression tests — cognitive scenarios from design sessions."""

from __future__ import annotations

import pytest

from mind.runtime import MindRuntime
from mind.types import CostLevel, Cue, CueKind


@pytest.fixture
def rt() -> MindRuntime:
    return MindRuntime.load_seed()


def test_pattern_missing_circle(rt: MindRuntime) -> None:
    cue = Cue(
        kind=CueKind.VISUAL,
        value="quadrato+cerchio,triangolo+cerchio,rettangolo+",
    )
    result = rt.think(cue)
    assert result.pattern_fill == "cerchio"
    assert "PATTERN:gap→fill(cerchio)" in result.explanation_symbols


def test_lamp_low_cost_replace_bulb(rt: MindRuntime) -> None:
    cue = Cue(kind=CueKind.TEXT, value="lampadina non si accende")
    result = rt.think(cue)
    assert result.action is not None
    assert result.action.id == "replace_bulb"
    assert result.fragments[0].title.startswith("ho cambiato la lampadina")


def test_lamp_high_cost_prefers_test(rt: MindRuntime) -> None:
    cue = Cue(kind=CueKind.TEXT, value="lampadina non si accende")
    result = rt.think(cue, cost_override=CostLevel.HIGH)
    assert result.action is not None
    assert result.action.id == "test_bulb"


def test_wedding_not_from_lamp_cue(rt: MindRuntime) -> None:
    cue = Cue(kind=CueKind.TEXT, value="lampadina non si accende")
    result = rt.think(cue)
    titles = " ".join(f.title for f in result.fragments)
    assert "matrimonio" not in titles.lower()


def test_wedding_from_family_cue(rt: MindRuntime) -> None:
    cue = Cue(kind=CueKind.TEXT, value="matrimonio di mio fratello")
    result = rt.think(cue)
    titles = " ".join(f.title for f in result.fragments)
    assert "matrimonio" in titles.lower()


def test_resonance_same_sensation_different_action(rt: MindRuntime) -> None:
    cue_new = Cue(
        kind=CueKind.TEXT,
        value="cliente whatsapp diffidente preventivo",
        meta={"human": True},
    )
    result = rt.think(
        cue_new,
        resonate_with="cliente marzo diffidente whatsapp",
    )
    assert any("RESONATE:" in s for s in result.explanation_symbols)
    assert result.action is not None
    assert result.action.id == "ask_one_more"
    assert result.action.id != "block_client"


def test_emotion_off_for_lamp(rt: MindRuntime) -> None:
    cue = Cue(kind=CueKind.TEXT, value="lampadina non si accende")
    result = rt.think(cue)
    assert result.human_emotion_active is False
    assert "EMO:off" in result.explanation_symbols


def test_emotion_on_for_wa(rt: MindRuntime) -> None:
    cue = Cue(kind=CueKind.TEXT, value="cliente whatsapp preventivo")
    result = rt.think(cue)
    assert result.human_emotion_active is True
    assert "EMO:on" in result.explanation_symbols
