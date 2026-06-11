"""Full ORGANISM perceive → think → express loop."""

from mind.types import CostLevel
from organism.runtime import OrganismRuntime


def test_shapes_pattern_speech():
    org = OrganismRuntime.studio_assistant()
    thought, expr, _ = org.live(
        {"shapes": "quadrato+cerchio,triangolo+cerchio,rettangolo+"},
        output_modality="text",
    )
    assert thought.mind_result.pattern_fill == "cerchio"
    assert expr.text is not None
    assert "cerchio" in expr.text.text.lower() or "Manca" in expr.text.text


def test_lampadina_replace_bulb():
    org = OrganismRuntime.studio_assistant()
    thought, expr, _ = org.live({"text": "lampadina non si accende"}, output_modality="speech")
    assert thought.mind_result.action is not None
    assert thought.mind_result.action.id == "replace_bulb"
    assert expr.speech is not None
    assert "lampadina" in expr.speech.text.lower()


def test_lampadina_high_cost_test_first():
    org = OrganismRuntime.studio_assistant()
    thought, _, _ = org.live(
        {"text": "lampadina non si accende"},
        cost_override=CostLevel.HIGH,
    )
    assert thought.mind_result.action.id == "test_bulb"


def test_wa_resonance_ask_one_more():
    org = OrganismRuntime.studio_assistant()
    thought, expr, _ = org.live(
        {"text": "cliente whatsapp diffidente chiede preventivo braccio"},
        resonate_with="cliente marzo diffidente whatsapp",
        output_modality="speech",
    )
    assert thought.mind_result.action.id == "ask_one_more"
    assert expr.speech is not None
    assert "domanda" in expr.speech.text.lower() or "dettaglio" in expr.speech.text.lower()


def test_booking_emotion_on():
    org = OrganismRuntime.studio_assistant()
    thought, expr, _ = org.live({"text": "Ciao, vorrei prenotare per giovedì"}, output_modality="speech")
    assert thought.mind_result.human_emotion_active is True
    assert expr.speech.prosody.get("emotion") == "empathetic"


def test_full_modality_output():
    org = OrganismRuntime.studio_assistant()
    _, expr, _ = org.live({"text": "preventivo tattoo braccio"}, output_modality="full")
    assert expr.speech and expr.song and expr.text and expr.motion


def test_audio_tone_perception():
    org = OrganismRuntime.studio_assistant()
    sensory = org.perceive({"tone_hz": 440.0})
    assert sensory.audio is not None
    assert len(sensory.audio.spikes) >= 0


def test_text_lexicon_hits():
    org = OrganismRuntime.studio_assistant()
    sensory = org.perceive({"text": "la lampadina non si accende"})
    assert "bulb_malfunction" in sensory.text.lexicon_hits
