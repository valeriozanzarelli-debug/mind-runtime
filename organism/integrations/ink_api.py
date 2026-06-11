"""ink-api bridge — WhatsApp thread → ORGANISM live → reply."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from organism.runtime import OrganismRuntime


@dataclass
class WaMessage:
    thread_id: str
    text: str
    audio_b64: str | None = None
    from_user: str = ""
    meta: dict[str, Any] | None = None

    @classmethod
    def from_mock(cls, text: str, *, thread_id: str = "wa_test_1") -> WaMessage:
        return cls(thread_id=thread_id, text=text, from_user="mock_client")


@dataclass
class WaReply:
    thread_id: str
    text: str
    ssml: str | None = None
    action_id: str | None = None
    learning: dict[str, Any] | None = None


class InkApiBridge:
    """
    Connect ORGANISM to ink-api dash endpoints.
    Mock mode works offline; live mode needs base_url + x-dash-token.
    """

    def __init__(
        self,
        base_url: str = "https://api.inkconscius.eu/dash/",
        token: str | None = None,
        *,
        mock: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.token = token
        self.mock = mock

    def fetch_thread_messages(self, thread_id: str, limit: int = 5) -> list[WaMessage]:
        if self.mock:
            return [WaMessage.from_mock("Ciao, preventivo braccio realistico?", thread_id=thread_id)]
        data = self._get(f"chat/threads/{thread_id}/messages?limit={limit}")
        return [_msg_from_api(m, thread_id) for m in data.get("messages", [])]

    def send_reply(self, thread_id: str, text: str) -> bool:
        if self.mock:
            return True
        self._post(
            f"chat/threads/{thread_id}/messages",
            {"body": text, "channel": "whatsapp"},
        )
        return True

    def live_from_message(
        self,
        organism: OrganismRuntime,
        message: WaMessage,
        *,
        output_modality: str = "speech",
        learn: bool = True,
    ) -> WaReply:
        input_data: dict[str, Any] = {"text": message.text}
        if message.audio_b64:
            import base64

            input_data["audio"] = base64.b64decode(message.audio_b64)
        thought, expr, learn_report = organism.live(
            input_data,
            output_modality=output_modality,  # type: ignore[arg-type]
            learn=learn,
        )
        text = ""
        ssml = None
        if expr.speech:
            text = expr.speech.text
            ssml = expr.speech.ssml
        elif expr.text:
            text = expr.text.text
        action_id = thought.mind_result.action.id if thought.mind_result.action else None
        if not self.mock and text:
            self.send_reply(message.thread_id, text)
        return WaReply(
            thread_id=message.thread_id,
            text=text,
            ssml=ssml,
            action_id=action_id,
            learning=learn_report.__dict__ if learn_report else None,
        )

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            self.base_url + path.lstrip("/"),
            headers=self._headers(),
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url + path.lstrip("/"),
            data=data,
            headers={**self._headers(), "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Accept": "application/json"}
        if self.token:
            h["x-dash-token"] = self.token
        return h


def _msg_from_api(raw: dict, thread_id: str) -> WaMessage:
    return WaMessage(
        thread_id=thread_id,
        text=str(raw.get("body") or raw.get("text") or ""),
        from_user=str(raw.get("from") or ""),
        meta=raw,
    )
