"""Text motor output — natural language from MindResult."""

from __future__ import annotations

from dataclasses import dataclass

from mind.types import MindResult


@dataclass
class TextOutput:
    text: str
    emotion: str
    action_id: str | None


class TextMotor:
    TEMPLATES = {
        "replace_bulb": "Nessun problema — la lampadina è quasi sempre quella. La sostituisco.",
        "test_bulb": "Prima verifico se la lampadina è davvero rotta, così non buttiamo soldi.",
        "ask_placement": "Certo! Mi dici dove sul braccio vorresti il tatuaggio?",
        "ask_one_more": "Capito. Prima di chiudere, mi serve un dettaglio in più — che stile hai in mente?",
        "send_quote": "Ti preparo il preventivo con i tempi e il costo.",
        "block_client": "Al momento non posso procedere con questa richiesta.",
    }

    def express(self, mind_result: MindResult, emotion: str = "calm") -> TextOutput:
        action_id = mind_result.action.id if mind_result.action else None
        if action_id and action_id in self.TEMPLATES:
            text = self.TEMPLATES[action_id]
        elif mind_result.pattern_fill:
            text = f"Manca: {mind_result.pattern_fill}"
        else:
            text = "Sto elaborando — dammi un attimo."
        return TextOutput(text=text, emotion=emotion, action_id=action_id)
