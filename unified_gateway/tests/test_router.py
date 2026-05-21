from __future__ import annotations

import pytest

from unified_gateway.app.config import Settings
from unified_gateway.app.providers.base import ProviderResult
from unified_gateway.app.router.engine import RoutingEngine
from unified_gateway.app.schemas import RouterOptions
from unified_gateway.app.state.rate_limit import RateLimitGuard


class FakeAdapter:
    def __init__(self, provider: str, ok: bool = True):
        self.provider = provider
        self.ok = ok
        self.calls: list[str] = []
        self.embedding_calls: list[str] = []

    async def chat_completions(self, payload, model=None):
        self.calls.append(str(model or ""))
        return ProviderResult(
            provider=self.provider,
            capability="chat.completions",
            ok=self.ok,
            status_code=200 if self.ok else 500,
            latency_ms=12.0,
            payload={"provider": self.provider, "model": model, "echo": payload},
            model=model or "m",
            error="" if self.ok else "failed",
        )

    async def embeddings(self, payload, model=None):
        self.embedding_calls.append(str(model or ""))
        return ProviderResult(
            provider=self.provider,
            capability="embeddings",
            ok=self.ok,
            status_code=200 if self.ok else 500,
            latency_ms=12.0,
            payload={"provider": self.provider, "model": model, "echo": payload},
            model=model or "m",
            error="" if self.ok else "failed",
        )


class FakeRegistry:
    def __init__(self):
        self.groq = FakeAdapter("groq", ok=False)
        self.gemini = FakeAdapter("gemini", ok=True)
        self.providers = {"groq": self.groq, "gemini": self.gemini}

    def names(self):
        return list(self.providers.keys())


@pytest.mark.asyncio
async def test_fallback_chain_moves_to_next_provider():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    engine = RoutingEngine(settings, FakeRegistry(), guard)

    options = RouterOptions(
        providers=["groq", "gemini"],
        models=["groq/llama-3.3-70b-versatile", "gemini/gemini-2.5-flash"],
        strategy="fallback_chain",
    )
    result = await engine.dispatch(
        capability="chat.completions",
        payload={"messages": [{"role": "user", "content": "hi"}]},
        options=options,
    )

    assert result["ok"] is True
    assert result["winner"]["provider"] == "gemini"
    assert len(result["results"]) == 2

    await guard.close()


@pytest.mark.asyncio
async def test_aggregate_returns_all():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    engine = RoutingEngine(settings, FakeRegistry(), guard)

    options = RouterOptions(
        providers=["groq", "gemini"],
        strategy="aggregate",
    )
    result = await engine.dispatch(
        capability="chat.completions",
        payload={"messages": [{"role": "user", "content": "hi"}]},
        options=options,
    )

    assert result["strategy"] == "aggregate"
    assert len(result["results"]) >= 1

    await guard.close()


@pytest.mark.asyncio
async def test_model_object_list_uses_priority_for_fallback_order():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    engine = RoutingEngine(settings, registry, guard)

    # groq has higher priority (0) and should be attempted first.
    payload = {
        "model": [
            {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0},
            {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    options = RouterOptions(strategy="fallback_chain", mode="latency_first")
    result = await engine.dispatch("chat.completions", payload, options)

    assert result["ok"] is True
    assert result["winner"]["provider"] == "gemini"
    # First call must be to groq model (priority 0).
    assert registry.groq.calls == ["llama-3.3-70b-versatile"]
    assert registry.gemini.calls == ["gemini-2.5-flash"]

    await guard.close()


@pytest.mark.asyncio
async def test_model_object_list_equal_priority_can_start_with_any_but_uses_all_on_failure():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    engine = RoutingEngine(settings, registry, guard)

    payload = {
        "model": [
            {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0},
            {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 0},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    options = RouterOptions(strategy="fallback_chain", mode="latency_first")
    result = await engine.dispatch("chat.completions", payload, options)

    assert result["ok"] is True
    assert {item["provider"] for item in result["results"]} == {"groq", "gemini"}
    assert result["winner"]["provider"] == "gemini"

    await guard.close()


@pytest.mark.asyncio
async def test_invalid_or_unknown_provider_entries_are_ignored_and_next_priority_used():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    engine = RoutingEngine(settings, registry, guard)

    payload = {
        "model": [
            {"provider": "unknown", "model": "x", "priority": -1},
            {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 2},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    options = RouterOptions(strategy="fallback_chain", mode="latency_first")
    result = await engine.dispatch("chat.completions", payload, options)

    assert result["ok"] is True
    assert result["winner"]["provider"] == "gemini"
    assert registry.groq.calls == []
    assert registry.gemini.calls == ["gemini-2.5-flash"]

    await guard.close()


@pytest.mark.asyncio
async def test_router_options_model_preferences_are_honored():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    engine = RoutingEngine(settings, registry, guard)

    options = RouterOptions(
        strategy="fallback_chain",
        mode="latency_first",
        model_preferences=[
            {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0},
            {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1},
        ],
    )
    result = await engine.dispatch(
        "chat.completions",
        {"messages": [{"role": "user", "content": "hi"}]},
        options,
    )

    assert result["ok"] is True
    assert result["winner"]["provider"] == "gemini"
    assert registry.groq.calls == ["llama-3.3-70b-versatile"]
    assert registry.gemini.calls == ["gemini-2.5-flash"]

    await guard.close()


@pytest.mark.asyncio
async def test_embeddings_default_candidates_use_embedding_models():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    engine = RoutingEngine(settings, registry, guard)

    result = await engine.dispatch("embeddings", {"input": "hello"}, RouterOptions(strategy="fallback_chain"))

    assert result["ok"] is True
    assert result["winner"]["provider"] == "gemini"
    assert registry.gemini.embedding_calls == ["gemini-embedding-001"]
    assert registry.groq.embedding_calls == []

    await guard.close()
