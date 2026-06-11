from organism.cognition.causal_graph import CausalGraph
from organism.cognition.comprehension import PsycheEngine
from organism.cognition.narrative import compose_narrative, relate_clauses
from organism.teaching.semantic_knowledge import SemanticKnowledge
from organism.teaching.story_curriculum import pinocchio_semantic_lessons


def _pinocchio_sem() -> SemanticKnowledge:
    sem = SemanticKnowledge()
    lessons = pinocchio_semantic_lessons()
    for word, definition, related in lessons["words"]:
        sem.teach_word(word, definition, related=related, story_id="pinocchio")
    for order, summary, entities, hooks in lessons["beats"]:
        sem.teach_beat("pinocchio", order, summary, entities=entities, hooks=hooks)
    return sem


def test_causal_lookup():
    g = CausalGraph()
    g.teach("perché piove", "quando l'acqua nell'aria diventa pesante, cade come pioggia.")
    hit = g.lookup("perché piove")
    assert hit is not None
    assert "pioggia" in hit.effect


def test_psyche_social_greeting():
    psyche = PsycheEngine()
    sem = _pinocchio_sem()
    pairs: dict[str, str] = {}

    def respond(heard: str):
        return pairs.get(heard.lower(), (None, None, True))

    pairs["ciao"] = ("ciao, ti ascolto. dimmi pure.", "speech", True)
    frame = psyche.comprehend("ciao", semantic=sem, dialogue_respond=respond)
    assert frame.intent == "social"
    assert "ascolto" in frame.taught_say
    assert frame.inhibit_lexicon_dump


def test_psyche_narrative_identity_focus():
    psyche = PsycheEngine()
    sem = _pinocchio_sem()
    frame = psyche.comprehend(
        "chi è pinocchio",
        semantic=sem,
        dialogue_respond=lambda h: (None, None, True),
    )
    assert frame.intent == "narrative_identity"
    assert frame.focus_entity == "pinocchio"
    assert len(frame.beats) == 1
    assert frame.narrative_max_beats == 1


def test_psyche_causal():
    psyche = PsycheEngine()
    sem = _pinocchio_sem()
    frame = psyche.comprehend(
        "perché piove",
        semantic=sem,
        dialogue_respond=lambda h: (None, None, True),
    )
    assert frame.intent == "causal"
    assert "pioggia" in frame.taught_say.lower()


def test_psyche_smalltalk_not_narrative():
    psyche = PsycheEngine()
    sem = _pinocchio_sem()
    frame = psyche.comprehend(
        "come stai",
        semantic=sem,
        dialogue_respond=lambda h: (None, None, True),
        wm_context=["pinocchio", "legno", "catasta", "bottega"],
    )
    assert frame.intent == "social"
    assert frame.inhibit_lexicon_dump
    assert "pinocchio" not in (frame.taught_say or "").lower()


def test_psyche_word_meaning():
    psyche = PsycheEngine()
    sem = _pinocchio_sem()
    sem.teach_word(
        "formaggio",
        "alimento dal latte, spesso giallo o bianco",
        related=["latte"],
        story_id="",
    )
    frame = psyche.comprehend(
        "formaggio",
        semantic=sem,
        dialogue_respond=lambda h: (None, None, True),
        wm_context=["pinocchio", "legno"],
    )
    assert frame.intent == "word_meaning"
    assert frame.focus_entity == "formaggio"
    assert "pinocchio" not in frame.themes


def test_psyche_explore_filters_story_contamination():
    psyche = PsycheEngine()
    sem = _pinocchio_sem()

    def episodic(_heard: str, limit: int = 3):
        return [
            {
                "heard": "raccontami pinocchio",
                "spoke": "Pinocchio legno poi bottega geppetto fra catasta.",
            }
        ]

    frame = psyche.comprehend(
        "come stai",
        semantic=sem,
        dialogue_respond=lambda h: (None, None, True),
        episodic_recall=episodic,
        wm_context=["pinocchio", "legno", "catasta"],
    )
    assert frame.intent == "social"
    assert not frame.episodic_hint


def test_parse_single_object_name():
    from organism.teaching.vision_phrase import is_vision_naming_phrase, parse_vision_teaching

    p = parse_vision_teaching("valigia")
    assert p.has_object
    assert p.object_name == "valigia"
    assert is_vision_naming_phrase("valigia")
    assert not is_vision_naming_phrase("come stai")
    assert not is_vision_naming_phrase("fra")


def test_visual_bind_strong_teach_consolidates():
    from organism.cognition.visual_bind import VisualBinder

    vb = VisualBinder()
    feat = {"color": "marrone", "scene_type": "oggetto", "object_sig": "abc123"}
    vb.bind(
        "sig1",
        "vedo un valigia marrone",
        boost=1.4,
        features=feat,
        object_sig="abc123",
        object_name="valigia",
        user_taught=True,
    )
    assert "valigia" in vb._user_taught
    assert vb.recognize_object(feat, object_sig="abc123", min_sim=0.38) == "valigia"


def test_relate_clauses():
    text = relate_clauses(
        ["Legno materiale dagli alberi", "Bottega piccolo laboratorio"],
        focus_entity="legno",
    )
    assert "poi" in text.lower()
    assert text.endswith(".")
