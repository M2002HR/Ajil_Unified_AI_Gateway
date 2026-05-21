from __future__ import annotations

from fastapi.testclient import TestClient

from unified_gateway.app.config import get_settings
from unified_gateway.app.main import app


def _setup_env(monkeypatch):
    monkeypatch.setenv("UAG_AUTH_ENABLED", "false")
    monkeypatch.setenv("UAG_ADMIN_ENABLED", "false")
    monkeypatch.setenv("UAG_LOG_ENABLED", "true")
    get_settings.cache_clear()


def test_chat_completions_stream_endpoint_returns_sse(monkeypatch):
    _setup_env(monkeypatch)
    with TestClient(app) as client:
        async def _fake_dispatch_chat_stream(payload, options):  # noqa: ANN001
            async def _iter():
                yield b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}\n\n'
                yield b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                yield b"data: [DONE]\n\n"

            return {
                "ok": True,
                "status_code": 200,
                "strategy": "fallback_chain",
                "mode": "limit_safe",
                "winner": {"provider": "gemini", "model": "gemini-2.5-flash"},
                "results": [],
                "stream": _iter(),
            }

        client.app.state.ctx.router.dispatch_chat_stream = _fake_dispatch_chat_stream
        resp = client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hello"}]},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert 'data: {"id":"x"' in resp.text
        assert "data: [DONE]" in resp.text
