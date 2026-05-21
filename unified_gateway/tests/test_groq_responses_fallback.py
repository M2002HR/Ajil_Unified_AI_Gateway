from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from unified_gateway.app.config import GroqProviderConfig
from unified_gateway.app.providers.groq_adapter import GroqAdapter


@dataclass
class _FakeResp:
    status_code: int
    headers: Dict[str, str]
    json_payload: Dict[str, Any] | None = None
    content: bytes = b""
    text: str = ""

    def json(self) -> Dict[str, Any]:
        if self.json_payload is None:
            raise ValueError("no json payload")
        return self.json_payload


class _FakeService:
    def __init__(self, responses: List[_FakeResp]) -> None:
        self.responses = responses
        self.calls: List[Dict[str, Any]] = []

    async def forward_json(self, *, path: str, payload: Dict[str, Any]) -> _FakeResp:
        self.calls.append({"path": path, "payload": dict(payload)})
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_groq_responses_falls_back_to_chat_completions_when_unsupported(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:2080")
    monkeypatch.setenv("all_proxy", "socks5://127.0.0.1:2080")
    adapter = GroqAdapter(GroqProviderConfig(api_keys=["dummy"]))
    adapter.service = _FakeService(
        responses=[
            _FakeResp(
                status_code=404,
                headers={"content-type": "application/json"},
                json_payload={"error": {"message": "responses not available"}},
            ),
            _FakeResp(
                status_code=200,
                headers={"content-type": "application/json"},
                json_payload={"id": "chatcmpl_1", "choices": [{"message": {"content": "ok"}}]},
            ),
        ]
    )

    result = await adapter.responses(
        payload={"input": "hello from responses"},
        model="llama-3.3-70b-versatile",
    )

    assert result.ok is True
    assert result.status_code == 200
    assert result.headers.get("x-uag-fallback") == "groq.responses->chat.completions"
    assert len(adapter.service.calls) == 2
    assert adapter.service.calls[0]["path"] == "/responses"
    assert adapter.service.calls[1]["path"] == "/chat/completions"
    assert adapter.service.calls[1]["payload"]["messages"][0]["content"] == "hello from responses"
