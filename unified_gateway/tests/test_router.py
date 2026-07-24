from __future__ import annotations

import asyncio

import pytest

from unified_gateway.app.config import Settings
from unified_gateway.app.providers.base import ProviderResult, ProviderStreamResult
from unified_gateway.app.router.engine import RoutingEngine
from unified_gateway.app.schemas import RouterOptions
from unified_gateway.app.state.rate_limit import RateLimitGuard


class FakeAdapter:
    def __init__(self, provider: str, ok: bool = True):
        self.provider = provider
        self.ok = ok
        self.calls: list[str] = []
        self.embedding_calls: list[str] = []
        self.stream_delay_sec: float = 0.0

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

    async def chat_completions_stream(self, payload, model=None, *, timeout_sec=None):
        if not self.ok:
            return ProviderStreamResult(
                provider=self.provider,
                capability="chat.completions",
                ok=False,
                status_code=504,
                latency_ms=12.0,
                model=model or "m",
                error="timeout",
            )
        if self.stream_delay_sec > 0:
            await asyncio.sleep(self.stream_delay_sec)

        async def _iter():
            yield b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}\n\n'
            yield b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"

        return ProviderStreamResult(
            provider=self.provider,
            capability="chat.completions",
            ok=True,
            status_code=200,
            latency_ms=12.0,
            model=model or "m",
            headers={},
            stream=_iter(),
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


class SlowAdapter(FakeAdapter):
    async def chat_completions(self, payload, model=None):
        await asyncio.sleep(1.2)
        return await super().chat_completions(payload, model=model)

    async def responses(self, payload, model=None):
        await asyncio.sleep(1.2)
        return ProviderResult(
            provider=self.provider,
            capability="responses",
            ok=False,
            status_code=500,
            latency_ms=0.0,
            payload=None,
            model=model or "m",
            error="slow",
        )


class SlowRegistry:
    def __init__(self):
        self.groq = SlowAdapter("groq", ok=False)
        self.gemini = SlowAdapter("gemini", ok=False)
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


@pytest.mark.asyncio
async def test_failure_summary_timeout_is_not_reported_as_429_when_strict_mode_enabled():
    settings = Settings()
    settings.router.strict_rate_limit_errors_only = True
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    engine = RoutingEngine(settings, FakeRegistry(), guard)

    failed = [
        ProviderResult(provider="groq", capability="chat.completions", ok=False, status_code=504, latency_ms=0.0, payload=None, error="timeout", model="m1"),
        ProviderResult(provider="gemini", capability="chat.completions", ok=False, status_code=504, latency_ms=0.0, payload=None, error="timeout", model="m2"),
    ]
    meta = engine._summarize_failure(failed)

    assert meta["status_code"] == 504
    assert meta["error_type"] == "upstream_timeout"
    assert meta["all_rate_limited"] is False
    await guard.close()


@pytest.mark.asyncio
async def test_failure_summary_all_429_is_reported_as_all_rate_limited():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    engine = RoutingEngine(settings, FakeRegistry(), guard)

    failed = [
        ProviderResult(provider="groq", capability="chat.completions", ok=False, status_code=429, latency_ms=0.0, payload=None, error="rate", model="m1"),
        ProviderResult(provider="gemini", capability="chat.completions", ok=False, status_code=429, latency_ms=0.0, payload=None, error="rate", model="m2"),
    ]
    meta = engine._summarize_failure(failed)

    assert meta["status_code"] == 429
    assert meta["error_type"] == "all_rate_limited"
    assert meta["all_rate_limited"] is True
    await guard.close()


@pytest.mark.asyncio
async def test_structured_output_never_uses_prose_local_fallback() -> None:
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    registry.groq.ok = False
    registry.gemini.ok = False
    engine = RoutingEngine(settings, registry, guard)

    result = await engine.dispatch(
        "chat.completions",
        {
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"},
        },
        RouterOptions(
            providers=["groq", "gemini"],
            strategy="fallback_chain",
            max_attempts=2,
        ),
    )

    assert result["ok"] is False
    assert result["winner"] is None
    assert all(item["provider"] != "local-fallback" for item in result["results"])
    await guard.close()


@pytest.mark.asyncio
async def test_chat_rescue_fallback_succeeds_even_if_primary_provider_fails():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    # force primary provider failure; gemini remains healthy in FakeRegistry
    registry.groq.ok = False
    registry.gemini.ok = True
    engine = RoutingEngine(settings, registry, guard)

    options = RouterOptions(
        providers=["groq"],
        strategy="fallback_chain",
        mode="limit_safe",
        timeout_sec=5,
        max_attempts=2,
    )
    payload = {
        "model": [{"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = await engine.dispatch("chat.completions", payload, options)

    assert result["ok"] is True
    assert result["winner"]["provider"] == "gemini"
    assert registry.groq.calls == ["llama-3.3-70b-versatile"]
    await guard.close()


@pytest.mark.asyncio
async def test_fallback_chain_respects_attempt_budget_without_unbounded_retries():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    registry.groq.ok = False
    engine = RoutingEngine(settings, registry, guard)

    options = RouterOptions(
        providers=["groq"],
        strategy="fallback_chain",
        mode="latency_first",
        timeout_sec=5,
        max_attempts=10,
    )
    result = await engine.dispatch("embeddings", {"input": "hello", "model": [{"provider": "groq", "model": "text-embedding-3-large", "priority": 0}]}, options)

    assert result["ok"] is False
    assert len(registry.groq.embedding_calls) == 1
    await guard.close()


@pytest.mark.asyncio
async def test_incompatible_chat_model_is_filtered_and_replaced_by_safe_default():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    registry.groq.ok = False
    registry.gemini.ok = True
    engine = RoutingEngine(settings, registry, guard)

    options = RouterOptions(
        providers=["groq"],
        strategy="fallback_chain",
        mode="limit_safe",
        timeout_sec=6,
        max_attempts=3,
    )
    payload = {
        "model": [{"provider": "groq", "model": "meta-llama/llama-prompt-guard-2-86m", "priority": 0}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = await engine.dispatch("chat.completions", payload, options)

    assert result["ok"] is True
    assert result["winner"]["provider"] == "gemini"
    await guard.close()


@pytest.mark.asyncio
async def test_parallel_race_timeout_populates_results_instead_of_no_attempt():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    engine = RoutingEngine(settings, SlowRegistry(), guard)

    options = RouterOptions(
        providers=["groq", "gemini"],
        strategy="parallel_race",
        mode="latency_first",
        timeout_sec=0.05,
        max_attempts=2,
    )
    result = await engine.dispatch("chat.completions", {"messages": [{"role": "user", "content": "hi"}]}, options)

    assert result["ok"] is True
    assert result["status_code"] == 200
    assert (result["winner"] or {}).get("provider") == "local-fallback"
    assert any(int(r.get("status_code", 0)) == 504 for r in result["results"])
    assert len(result["results"]) >= 1
    await guard.close()


@pytest.mark.asyncio
async def test_aggregate_timeout_populates_results_instead_of_no_attempt():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    engine = RoutingEngine(settings, SlowRegistry(), guard)

    options = RouterOptions(
        providers=["groq", "gemini"],
        strategy="aggregate",
        mode="latency_first",
        timeout_sec=0.05,
        max_attempts=2,
    )
    result = await engine.dispatch("responses", {"messages": [{"role": "user", "content": "hi"}]}, options)

    assert result["ok"] is True
    assert result["status_code"] == 200
    assert (result["winner"] or {}).get("provider") == "local-fallback"
    assert any(int(r.get("status_code", 0)) == 504 for r in result["results"])
    assert len(result["results"]) >= 1
    await guard.close()


@pytest.mark.asyncio
async def test_local_fallback_returns_success_for_chat_when_upstreams_timeout():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    engine = RoutingEngine(settings, SlowRegistry(), guard)

    options = RouterOptions(
        providers=["groq", "gemini"],
        strategy="parallel_race",
        mode="latency_first",
        timeout_sec=0.05,
        max_attempts=2,
    )
    result = await engine.dispatch("chat.completions", {"messages": [{"role": "user", "content": "سلام"}]}, options)

    assert result["ok"] is True
    assert (result["winner"] or {}).get("provider") == "local-fallback"
    assert int(result.get("status_code", 500)) == 200
    await guard.close()


@pytest.mark.asyncio
async def test_local_fallback_does_not_mask_all_rate_limited_failures():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    engine = RoutingEngine(settings, FakeRegistry(), guard)

    async def _deny(*args, **kwargs):
        return False

    engine.guard.allow = _deny  # type: ignore[assignment]

    options = RouterOptions(
        providers=["groq", "gemini"],
        strategy="fallback_chain",
        mode="limit_safe",
        timeout_sec=1,
        max_attempts=2,
    )
    result = await engine.dispatch("chat.completions", {"messages": [{"role": "user", "content": "hi"}]}, options)

    assert result["ok"] is False
    assert int(result.get("status_code", 0)) == 429
    assert bool(result.get("all_rate_limited")) is True
    await guard.close()


@pytest.mark.asyncio
async def test_chat_stream_fallback_moves_to_next_provider():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    registry.groq.ok = False
    registry.gemini.ok = True
    engine = RoutingEngine(settings, registry, guard)

    options = RouterOptions(
        providers=["groq", "gemini"],
        strategy="fallback_chain",
        mode="latency_first",
        timeout_sec=5,
        max_attempts=2,
    )
    result = await engine.dispatch_chat_stream(
        payload={
            "model": [
                {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0},
                {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1},
            ],
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        options=options,
    )
    assert result["ok"] is True
    assert (result["winner"] or {}).get("provider") == "gemini"
    assert len(result["results"]) == 2
    assert result.get("stream") is not None
    await guard.close()


@pytest.mark.asyncio
async def test_chat_stream_local_fallback_when_upstreams_fail():
    settings = Settings()
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    registry.groq.ok = False
    registry.gemini.ok = False
    engine = RoutingEngine(settings, registry, guard)
    options = RouterOptions(providers=["groq", "gemini"], strategy="fallback_chain", timeout_sec=1, max_attempts=2)
    result = await engine.dispatch_chat_stream(
        payload={"messages": [{"role": "user", "content": "سلام"}], "stream": True},
        options=options,
    )
    assert result["ok"] is True
    assert (result["winner"] or {}).get("provider") == "local-fallback"
    assert result.get("stream") is not None
    await guard.close()


@pytest.mark.asyncio
async def test_chat_stream_uses_request_timeout_budget_not_fixed_attempt_cap():
    settings = Settings()
    settings.router.fallback_attempt_timeout_sec = 0.05
    guard = RateLimitGuard(redis_url="redis://127.0.0.1:6379/9", key_prefix="test", required=False)
    registry = FakeRegistry()
    registry.groq.ok = True
    registry.groq.stream_delay_sec = 0.12
    engine = RoutingEngine(settings, registry, guard)

    options = RouterOptions(
        providers=["groq"],
        strategy="fallback_chain",
        mode="latency_first",
        timeout_sec=0.3,
        max_attempts=1,
    )
    result = await engine.dispatch_chat_stream(
        payload={
            "model": [{"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0}],
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        options=options,
    )
    assert result["ok"] is True
    assert (result["winner"] or {}).get("provider") == "groq"
    assert result.get("stream") is not None
    await guard.close()
