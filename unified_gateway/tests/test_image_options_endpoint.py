from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from unified_gateway.app.config import Settings
from unified_gateway.app.main import app
from unified_gateway.app.state.event_tracker import EventTracker
import unified_gateway.app.main as main_module


@pytest.mark.asyncio
async def test_image_options_endpoint(monkeypatch):
    settings = Settings()
    settings.auth.enabled = True
    settings.auth.token = "client-token"
    settings.auth.header_name = "x-api-token"

    fake_ctx = SimpleNamespace(settings=settings, event_tracker=EventTracker(max_events=100))
    monkeypatch.setattr(main_module, "_ctx", lambda request: fake_ctx)

    async def fake_resolve(ctx, *, provider: str, model: str, refresh: bool = False):
        assert ctx is fake_ctx
        assert provider == "pollinations"
        assert model == "flux"
        return {
            "provider": provider,
            "model": model,
            "sizes": ["512x512", "1024x1024"],
            "qualities": ["low", "medium", "high"],
            "default_size": "1024x1024",
            "default_quality": "medium",
            "source": "probe+raw",
            "fetched_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(main_module, "_resolve_image_options", fake_resolve)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.get("/v1/images/options?provider=pollinations&model=flux")
        assert unauthorized.status_code == 401

        headers = {"x-api-token": "client-token"}
        resp = await client.get("/v1/images/options?provider=pollinations&model=flux", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "pollinations"
        assert data["model"] == "flux"
        assert data["default_size"] == "1024x1024"
        assert "medium" in data["qualities"]
