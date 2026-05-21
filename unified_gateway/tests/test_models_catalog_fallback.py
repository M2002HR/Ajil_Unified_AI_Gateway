from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from unified_gateway.app.config import Settings
import unified_gateway.app.main as main_module


class _FailAdapter:
    async def list_models(self):
        raise RuntimeError("upstream unavailable")


@pytest.mark.asyncio
async def test_refresh_models_catalog_uses_static_fallback_when_all_providers_fail():
    settings = Settings()
    ctx = SimpleNamespace(
        settings=settings,
        registry=SimpleNamespace(providers={"gemini": _FailAdapter(), "groq": _FailAdapter(), "pollinations": _FailAdapter()}),
        models_catalog_cache={"fetched_at": "", "ttl_sec": 180, "provider_rows": [], "items": []},
        models_catalog_lock=asyncio.Lock(),
    )

    result = await main_module._refresh_models_catalog(ctx, refresh=True, timeout_sec=0.1)

    assert result["fallback_applied"] is True
    assert len(result["items"]) > 0

    caps = {cap for row in result["items"] for cap in row.get("capabilities", [])}
    assert "chat.completions" in caps
    assert "responses" in caps
    assert "embeddings" in caps
    assert "images.generations" in caps
    assert "audio.speech" in caps
    assert "audio.transcriptions" in caps
