from organism.cognition.narrative import compose_narrative, narrative_quality
from organism.teaching.semantic_knowledge import SemanticKnowledge, is_narrative_request
from organism.teaching.story_curriculum import pinocchio_semantic_lessons

PINOCCHIO_OPENING = (
    "C'era una volta un pezzo di legno. Non era un legno di lusso, ma un semplice pezzo da catasta"
)


def test_narrative_request():
    assert is_narrative_request("raccontami pinocchio")
    assert is_narrative_request("chi è pinocchio")
    assert is_narrative_request("raccontami la favola del corvo")
    assert not is_narrative_request("perché piove")


def test_grounded_words_only_in_recall():
    sem = SemanticKnowledge()
    sem.teach_word("legno", "materiale dagli alberi", story_id="pinocchio")
    sem.teach_beat(
        "pinocchio",
        1,
        "c'era un pezzo di legno",
        entities=["legno", "falegname"],
        hooks=["legno"],
    )
    themes = sem.recall_story_themes("raccontami pinocchio")
    assert "legno" in themes
    assert "falegname" not in themes


def test_narrate_plan_is_not_opening_verbatim():
    sem = SemanticKnowledge()
    lessons = pinocchio_semantic_lessons()
    for word, definition, related in lessons["words"]:
        sem.teach_word(word, definition, related=related, story_id="pinocchio")
    for order, summary, entities, hooks in lessons["beats"]:
        sem.teach_beat("pinocchio", order, summary, entities=entities, hooks=hooks)

    plan = sem.narrate_plan("raccontami pinocchio")
    joined = " ".join(plan)
    assert PINOCCHIO_OPENING not in joined
    assert "legno" in joined
    assert sem.coverage("pinocchio")["ratio"] == 1.0


def test_narrate_beats_ordered():
    sem = SemanticKnowledge()
    lessons = pinocchio_semantic_lessons()
    for word, definition, related in lessons["words"]:
        sem.teach_word(word, definition, related=related, story_id="pinocchio")
    for order, summary, entities, hooks in lessons["beats"]:
        sem.teach_beat("pinocchio", order, summary, entities=entities, hooks=hooks)

    beats = sem.narrate_beats("raccontami pinocchio")
    assert len(beats) == 4
    assert beats[0].order == 1
    assert beats[-1].entities[0] == "pinocchio"


def test_compose_narrative_from_definitions_not_verbatim():
    sem = SemanticKnowledge()
    lessons = pinocchio_semantic_lessons()
    for word, definition, related in lessons["words"]:
        sem.teach_word(word, definition, related=related, story_id="pinocchio")
    for order, summary, entities, hooks in lessons["beats"]:
        sem.teach_beat("pinocchio", order, summary, entities=entities, hooks=hooks)

    beats = sem.narrate_beats("raccontami pinocchio")
    text = compose_narrative(
        beats,
        definition_fn=sem.definition,
        grounded_fn=sem.is_grounded,
        articulable_fn=lambda w: True,
    )
    assert PINOCCHIO_OPENING not in text
    assert "legno" in text.lower()
    assert "pinocchio" in text.lower()
    assert "burattino" in text.lower()
    assert "penso" not in text.lower()
    assert "raccontami" not in text.lower()
    assert narrative_quality(text, expected_entities=["legno", "pinocchio", "burattino"]) >= 0.5


def test_compose_narrative_no_discourse_openers():
    sem = SemanticKnowledge()
    lessons = pinocchio_semantic_lessons()
    for word, definition, related in lessons["words"]:
        sem.teach_word(word, definition, related=related, story_id="pinocchio")
    for order, summary, entities, hooks in lessons["beats"]:
        sem.teach_beat("pinocchio", order, summary, entities=entities, hooks=hooks)

    text = compose_narrative(
        sem.narrate_beats("raccontami pinocchio"),
        definition_fn=sem.definition,
        grounded_fn=sem.is_grounded,
        articulable_fn=lambda w: True,
    )
    for bad in ("penso", "noto", "osservo", "raccontami", "credo che"):
        assert bad not in text.lower()
