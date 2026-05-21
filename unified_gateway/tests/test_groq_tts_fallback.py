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
async def test_tts_fallback_from_decommissioned_playai_model(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:2080")
    monkeypatch.setenv("all_proxy", "socks5://127.0.0.1:2080")
    adapter = GroqAdapter(GroqProviderConfig(api_keys=["dummy"]))
    adapter.service = _FakeService(
        responses=[
            _FakeResp(
                status_code=400,
                headers={"content-type": "application/json"},
                json_payload={
                    "error": {
                        "message": "The model `playai-tts` has been decommissioned and is no longer supported.",
                        "code": "model_decommissioned",
                    }
                },
            ),
            _FakeResp(
                status_code=200,
                headers={"content-type": "audio/wav"},
                content=b"RIFF....WAVE",
            ),
        ]
    )

    result = await adapter.tts(
        payload={"input": "hello", "voice": "Fritz-PlayAI"},
        model="playai-tts",
    )

    assert result.ok is True
    assert result.status_code == 200
    assert result.model == "canopylabs/orpheus-v1-english"
    assert result.headers.get("x-uag-tts-fallback-from") == "playai-tts"
    assert result.headers.get("x-uag-tts-fallback-to") == "canopylabs/orpheus-v1-english"
    assert len(adapter.service.calls) == 2
    assert adapter.service.calls[0]["payload"]["model"] == "playai-tts"
    assert adapter.service.calls[1]["payload"]["model"] == "canopylabs/orpheus-v1-english"
