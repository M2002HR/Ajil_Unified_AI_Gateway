from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from unified_gateway.app.config import Settings
from unified_gateway.app.state.event_tracker import EventTracker
from unified_gateway.app.main import app
import unified_gateway.app.main as main_module


@pytest.mark.asyncio
async def test_models_catalog_endpoints(monkeypatch):
    settings = Settings()
    settings.auth.enabled = True
    settings.auth.token = "client-token"
    settings.auth.header_name = "x-api-token"

    fake_ctx = SimpleNamespace(settings=settings, event_tracker=EventTracker(max_events=100))
    monkeypatch.setattr(main_module, "_ctx", lambda request: fake_ctx)

    async def fake_refresh(ctx, *, refresh: bool, timeout_sec: float = 30.0):
        assert ctx is fake_ctx
        return {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "from_cache": True,
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "provider_rows": [
                {"provider": "gemini", "ok": True, "status_code": 200, "latency_ms": 10.0, "error": "", "count": 1, "items": []},
                {"provider": "pollinations", "ok": True, "status_code": 200, "latency_ms": 12.0, "error": "", "count": 1, "items": []},
            ],
            "items": [
                {
                    "provider": "gemini",
                    "id": "gemma-4-27b-it",
                    "label": "Gemma 4",
                    "family": "gemma",
                    "capabilities": ["chat.completions", "responses"],
                    "model_type": "llm",
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "preview": False,
                    "paid_only": False,
                    "raw": {"name": "models/gemma-4-27b-it"},
                },
                {
                    "provider": "pollinations",
                    "id": "flux",
                    "label": "Flux",
                    "family": "flux",
                    "capabilities": ["images.generations"],
                    "model_type": "image",
                    "input_modalities": ["text"],
                    "output_modalities": ["image"],
                    "preview": False,
                    "paid_only": False,
                    "raw": {"name": "flux"},
                },
            ],
        }

    monkeypatch.setattr(main_module, "_refresh_models_catalog", fake_refresh)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.get("/v1/models/catalog")
        assert unauthorized.status_code == 401

        headers = {"x-api-token": "client-token"}

        resp = await client.get(
            "/v1/models/catalog?providers=gemini&capability=chat.completions&include_raw=false",
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["items"][0]["provider"] == "gemini"
        assert "raw" not in data["items"][0]

        image_only = await client.get("/v1/models/catalog?model_type=image", headers=headers)
        assert image_only.status_code == 200
        assert image_only.json()["count"] == 1
        assert image_only.json()["items"][0]["provider"] == "pollinations"

        summary = await client.get("/v1/models/catalog/summary", headers=headers)
        assert summary.status_code == 200
        summary_data = summary.json()
        assert summary_data["summary"]["total"] == 2

        providers = await client.get("/v1/models/providers", headers=headers)
        assert providers.status_code == 200
        providers_data = providers.json()
        assert providers_data["count"] == 2
