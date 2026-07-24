from __future__ import annotations

import httpx
import pytest

from modules.groq_proxy.api.app.config import Settings
from modules.groq_proxy.api.app.services import GroqProxyService


def _settings() -> Settings:
    settings = Settings()
    settings.groq.api_keys = ["test-key-one", "test-key-two"]
    settings.proxy.retry_on_429 = True
    settings.proxy.retry_on_5xx = True
    settings.proxy.max_retries_per_key = 2
    settings.proxy.max_retries_on_5xx = 2
    settings.proxy.retry_backoff_sec = 0
    settings.proxy.cooloff_sec = 0
    return settings


@pytest.mark.asyncio
async def test_rate_limited_key_is_cooled_and_next_key_can_succeed() -> None:
    service = GroqProxyService(_settings())
    calls: list[str] = []

    async def fake_request(**kwargs: object) -> httpx.Response:
        headers = kwargs["headers"]
        assert isinstance(headers, dict)
        key = str(headers["Authorization"])
        calls.append(key)
        request = httpx.Request("POST", "https://api.example.test/chat/completions")
        if key.endswith("one"):
            return httpx.Response(429, headers={"retry-after": "3"}, request=request)
        return httpx.Response(200, json={"id": "ok"}, request=request)

    service.client.request = fake_request  # type: ignore[method-assign]
    response = await service.forward_json(path="/chat/completions", payload={"model": "test"})

    assert response.status_code == 200
    assert calls == ["Bearer test-key-one", "Bearer test-key-two"]
    assert response.headers["x-proxy-key-rotated"] == "true"
    assert response.headers["x-proxy-all-keys-rate-limited"] == "false"
    await service.aclose()


@pytest.mark.asyncio
async def test_returns_429_only_after_every_key_is_rate_limited() -> None:
    service = GroqProxyService(_settings())
    calls: list[str] = []

    async def fake_request(**kwargs: object) -> httpx.Response:
        headers = kwargs["headers"]
        assert isinstance(headers, dict)
        calls.append(str(headers["Authorization"]))
        return httpx.Response(
            429,
            headers={"x-ratelimit-reset-requests": "2.5s"},
            request=httpx.Request("POST", "https://api.example.test/chat/completions"),
        )

    service.client.request = fake_request  # type: ignore[method-assign]
    response = await service.forward_json(path="/chat/completions", payload={"model": "test"})

    assert response.status_code == 429
    assert calls == ["Bearer test-key-one", "Bearer test-key-two"]
    assert response.headers["x-proxy-all-keys-rate-limited"] == "true"
    assert response.headers["retry-after"] == "3"
    assert response.headers["x-proxy-attempts"] == "2"
    await service.aclose()


@pytest.mark.asyncio
async def test_server_error_rotates_to_another_key_before_returning_failure() -> None:
    service = GroqProxyService(_settings())
    calls: list[str] = []

    async def fake_request(**kwargs: object) -> httpx.Response:
        headers = kwargs["headers"]
        assert isinstance(headers, dict)
        key = str(headers["Authorization"])
        calls.append(key)
        request = httpx.Request("POST", "https://api.example.test/chat/completions")
        if key.endswith("one"):
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"id": "ok"}, request=request)

    service.client.request = fake_request  # type: ignore[method-assign]
    response = await service.forward_json(path="/chat/completions", payload={"model": "test"})

    assert response.status_code == 200
    assert calls == ["Bearer test-key-one", "Bearer test-key-two"]
    await service.aclose()


@pytest.mark.asyncio
async def test_server_failures_preserve_the_upstream_status_after_bounded_rotation() -> None:
    settings = _settings()
    settings.proxy.max_retries_on_5xx = 1
    service = GroqProxyService(settings)
    calls: list[str] = []

    async def fake_request(**kwargs: object) -> httpx.Response:
        headers = kwargs["headers"]
        assert isinstance(headers, dict)
        calls.append(str(headers["Authorization"]))
        return httpx.Response(
            503,
            request=httpx.Request("POST", "https://api.example.test/chat/completions"),
        )

    service.client.request = fake_request  # type: ignore[method-assign]
    response = await service.forward_json(path="/chat/completions", payload={"model": "test"})

    assert response.status_code == 503
    assert calls == ["Bearer test-key-one", "Bearer test-key-two"]
    assert response.headers["x-proxy-all-keys-rate-limited"] == "false"
    await service.aclose()
