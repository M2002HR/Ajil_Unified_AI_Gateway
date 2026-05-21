from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from unified_gateway.app.config import Settings
from unified_gateway.app.main import app
import unified_gateway.app.main as main_module
from unified_gateway.app.providers.base import ProviderResult
from unified_gateway.app.router.engine import RoutingEngine
from unified_gateway.app.schemas import RouterOptions
from unified_gateway.app.state.event_tracker import EventTracker
from unified_gateway.app.state.rate_limit import RateLimitGuard
from unified_gateway.app.state.usage_tracker import UsageTracker


class _FakeAdapter:
    def __init__(self, provider: str, ok: bool):
        self.provider = provider
        self.ok = ok

    async def chat_completions(self, payload, model=None):
        response_payload = {
            "usage": {
                "prompt_tokens": 11 if self.provider == "groq" else 7,
                "completion_tokens": 5 if self.provider == "groq" else 3,
                "total_tokens": 16 if self.provider == "groq" else 10,
            }
        }
        return ProviderResult(
            provider=self.provider,
            capability="chat.completions",
            ok=self.ok,
            status_code=200 if self.ok else 429,
            latency_ms=9.0 if self.ok else 14.0,
            payload=response_payload,
            model=model or "",
            error="" if self.ok else "rate limited",
            headers={
                "x-proxy-key-mask": "****A1" if self.provider == "groq" else "****B2",
                "x-ratelimit-limit-requests": "1000",
                "x-ratelimit-remaining-requests": "997" if self.ok else "0",
                "retry-after": "2" if not self.ok else "0",
            },
        )


class _FakeRegistry:
    def __init__(self):
        self.providers = {
            "groq": _FakeAdapter("groq", ok=False),
            "gemini": _FakeAdapter("gemini", ok=True),
        }

    def names(self):
        return list(self.providers.keys())


@pytest.mark.asyncio
async def test_usage_tracker_records_and_aggregates_by_key_provider_model():
    tracker = UsageTracker(max_events=2000)

    await tracker.record(
        capability="chat.completions",
        priority=0,
        result=ProviderResult(
            provider="groq",
            capability="chat.completions",
            ok=False,
            status_code=429,
            latency_ms=23.4,
            payload={"usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}},
            error="rate limited",
            model="llama-3.3-70b-versatile",
            headers={
                "x-proxy-key-mask": "****A1",
                "x-ratelimit-limit-requests": "1000",
                "x-ratelimit-remaining-requests": "0",
                "retry-after": "2",
            },
        ),
    )
    await tracker.record(
        capability="chat.completions",
        priority=1,
        result=ProviderResult(
            provider="gemini",
            capability="chat.completions",
            ok=True,
            status_code=200,
            latency_ms=11.2,
            payload={"usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 6, "totalTokenCount": 14}},
            error="",
            model="gemini-2.5-flash",
            headers={"x-proxy-key-mask": "****B2"},
        ),
    )

    overview = await tracker.overview(since_minutes=60)
    assert overview["overall"]["requests_total"] == 2
    assert overview["overall"]["status_429"] == 1
    assert overview["overall"]["tokens_total"] == 24

    by_provider = await tracker.aggregate(group_by="provider", since_minutes=60)
    providers = {item["group"]: item for item in by_provider["items"]}
    assert providers["groq"]["requests_total"] == 1
    assert providers["gemini"]["requests_total"] == 1

    by_key = await tracker.aggregate(group_by="key", since_minutes=60)
    key_groups = {item["group"] for item in by_key["items"]}
    assert "groq:****A1" in key_groups
    assert "gemini:****B2" in key_groups

    limits = await tracker.key_limits_latest()
    assert limits["count"] >= 1


@pytest.mark.asyncio
async def test_routing_engine_fallback_records_all_attempts_with_priorities():
    settings = Settings()
    tracker = UsageTracker(max_events=2000)
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    engine = RoutingEngine(settings, _FakeRegistry(), guard, usage_tracker=tracker)

    payload = {
        "model": [
            {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0},
            {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1},
        ],
        "messages": [{"role": "user", "content": "hello"}],
    }
    result = await engine.dispatch("chat.completions", payload, RouterOptions(strategy="fallback_chain"))
    assert result["ok"] is True

    events = await tracker.events(since_minutes=60, limit=10)
    assert events["count"] == 2
    observed = {(item["provider"], item["priority"], item["status_code"]) for item in events["items"]}
    assert ("groq", 0, 429) in observed
    assert ("gemini", 1, 200) in observed

    await guard.close()


@pytest.mark.asyncio
async def test_admin_usage_endpoints_return_expected_shapes(monkeypatch):
    tracker = UsageTracker(max_events=1000)
    await tracker.record(
        capability="chat.completions",
        priority=0,
        result=ProviderResult(
            provider="groq",
            capability="chat.completions",
            ok=True,
            status_code=200,
            latency_ms=12.0,
            payload={"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
            model="llama-3.3-70b-versatile",
            headers={"x-proxy-key-mask": "****A1", "x-ratelimit-remaining-requests": "999"},
        ),
    )

    settings = Settings()
    settings.admin.enabled = True
    settings.admin.token = "admin-token"
    settings.admin.header_name = "x-admin-token"
    fake_ctx = SimpleNamespace(
        settings=settings,
        usage_tracker=tracker,
        event_tracker=EventTracker(max_events=100),
        router=SimpleNamespace(stats=lambda: {"items": []}),
    )
    monkeypatch.setattr(main_module, "_ctx", lambda request: fake_ctx)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauth = await client.get("/admin/usage/overview")
        assert unauth.status_code == 401

        headers = {"x-admin-token": "admin-token"}
        overview = await client.get("/admin/usage/overview?since_minutes=60", headers=headers)
        assert overview.status_code == 200
        assert overview.json()["overall"]["requests_total"] == 1

        by_provider = await client.get("/admin/usage/providers?since_minutes=60", headers=headers)
        assert by_provider.status_code == 200
        assert by_provider.json()["group_by"] == "provider"

        by_models = await client.get("/admin/usage/models?since_minutes=60", headers=headers)
        assert by_models.status_code == 200
        assert by_models.json()["group_by"] == "model"

        by_keys = await client.get("/admin/usage/keys?since_minutes=60", headers=headers)
        assert by_keys.status_code == 200
        assert by_keys.json()["group_by"] == "key"

        events = await client.get("/admin/usage/events?since_minutes=60&limit=10", headers=headers)
        assert events.status_code == 200
        assert events.json()["count"] == 1

        key_limits = await client.get("/admin/usage/key-limits/latest?provider=groq", headers=headers)
        assert key_limits.status_code == 200
        assert key_limits.json()["count"] == 1

        invalid = await client.get("/admin/usage/aggregate?group_by=invalid", headers=headers)
        assert invalid.status_code == 400
