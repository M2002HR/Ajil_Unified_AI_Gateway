from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from unified_gateway.app.config import GeminiProviderConfig
from unified_gateway.app.providers.gemini_adapter import GeminiAdapter


@dataclass
class _FakeResp:
    status_code: int
    payload: Dict[str, Any]
    headers: Dict[str, str]

    def json(self) -> Dict[str, Any]:
        return self.payload


class _FakeService:
    def __init__(self, responses: List[_FakeResp]) -> None:
        self.responses = responses
        self.calls: List[Dict[str, Any]] = []

    async def proxy_call(self, body: Dict[str, Any]) -> _FakeResp:
        self.calls.append(dict(body))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_gemini_embeddings_fallback_when_requested_model_not_found():
    adapter = GeminiAdapter(GeminiProviderConfig(api_keys=["dummy"], mode="gemini_direct"))
    adapter.service = _FakeService(
        responses=[
            _FakeResp(
                status_code=404,
                payload={
                    "error": {
                        "message": "models/text-embedding-004 is not found for API version v1beta, or is not supported for embedContent.",
                        "status": "NOT_FOUND",
                    }
                },
                headers={"content-type": "application/json"},
            ),
            _FakeResp(
                status_code=200,
                payload={"embedding": {"values": [0.1, 0.2]}},
                headers={"content-type": "application/json"},
            ),
        ]
    )

    result = await adapter.embeddings({"input": "hello"}, model="text-embedding-004")

    assert result.ok is True
    assert result.status_code == 200
    assert result.model == "gemini-embedding-001"
    assert len(adapter.service.calls) == 2
    assert adapter.service.calls[0]["model"] == "text-embedding-004"
    assert adapter.service.calls[1]["model"] == "gemini-embedding-001"

