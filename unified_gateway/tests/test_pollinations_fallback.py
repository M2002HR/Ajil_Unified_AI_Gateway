from __future__ import annotations

import pytest

from unified_gateway.app.config import PollinationsProviderConfig
from unified_gateway.app.providers.pollinations_adapter import PollinationsAdapter


@pytest.mark.asyncio
async def test_pollinations_paid_error_falls_back_to_public_image_route():
    cfg = PollinationsProviderConfig(
        enabled=True,
        api_keys=["test-key"],
        default_image_model="flux",
        trust_env_proxy=False,
    )
    adapter = PollinationsAdapter(cfg)

    async def fake_generate_image(payload):
        return (
            402,
            {
                "success": False,
                "error": {
                    "message": "Insufficient balance for this key",
                    "code": "PAYMENT_REQUIRED",
                },
            },
            {"x-proxy-attempts": "1"},
        )

    async def fake_list_free_image_models(force_refresh=False):
        return [{"name": "flux", "paid_only": False, "output_modalities": ["image"]}]

    async def fake_generate_image_get_public(prompt, params, request_id=None):
        return (200, b"\x89PNG\r\n\x1a\nabc", {"x-proxy-served-by": "public-image-route"}, "image/png")

    adapter.service.generate_image = fake_generate_image  # type: ignore[assignment]
    adapter.service.list_free_image_models = fake_list_free_image_models  # type: ignore[assignment]
    adapter.service.generate_image_get_public = fake_generate_image_get_public  # type: ignore[assignment]

    result = await adapter.image_generations(
        {
            "prompt": "minimal cat icon",
            "size": "512x512",
            "quality": "low",
            "response_format": "url",
        },
        model="flux",
    )

    assert result.ok is True
    assert result.status_code == 200
    assert result.payload["provider_fallback"]["route"] == "public-image"
    assert result.payload["data"][0]["b64_json"]
    assert result.headers.get("x-pollinations-fallback") == "public-image"

    await adapter.aclose()
