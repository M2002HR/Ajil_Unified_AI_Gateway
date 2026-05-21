from __future__ import annotations

from typing import Any, Dict

from modules.pollinations_proxy.api.app.config import (  # type: ignore
    AdminConfig,
    AppConfig,
    ImageDefaults,
    PollinationsConfig,
    Settings as PollinationsSettings,
)
from modules.pollinations_proxy.api.app.services import PollinationsProxyService  # type: ignore

from ..config import PollinationsProviderConfig
from .base import ProviderAdapter, ProviderResult


class PollinationsAdapter(ProviderAdapter):
    name = "pollinations"

    def __init__(self, cfg: PollinationsProviderConfig) -> None:
        keys = list(cfg.api_keys)
        if not keys:
            keys = ["__missing__"]

        settings = PollinationsSettings(
            app=AppConfig(request_timeout_sec=120),
            pollinations=PollinationsConfig(
                base_url=cfg.base_url,
                api_keys=keys,
                default_image_model=cfg.default_image_model,
                use_proxy_2080=cfg.use_proxy_2080,
                proxy_2080_url=cfg.proxy_2080_url,
                trust_env_proxy=cfg.trust_env_proxy,
                max_attempts_per_request=cfg.max_attempts_per_request,
                retry_status_codes=cfg.retry_status_codes,
                retry_backoff_sec=cfg.retry_backoff_sec,
                cooldown_sec=cfg.cooldown_sec,
            ),
            image_defaults=ImageDefaults(
                n=cfg.image_default_n,
                size=cfg.image_default_size,
                quality=cfg.image_default_quality,
                response_format=cfg.image_default_response_format,
            ),
            admin=AdminConfig(enabled=False),
        )
        self.service = PollinationsProxyService(settings)

    async def aclose(self) -> None:
        await self.service.close()

    async def image_generations(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model
        try:
            status, parsed, headers = await self.service.generate_image(body)
            return ProviderResult(self.name, "images.generations", status < 400, status, self.done(start), parsed, "" if status < 400 else str(parsed), str(body.get("model", "")), headers)
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "images.generations", False, 502, self.done(start), None, str(exc), str(body.get("model", "")))

    async def list_models(self) -> ProviderResult:
        start = self.started()
        try:
            models = await self.service.list_image_models(force_refresh=False)
            return ProviderResult(self.name, "models", True, 200, self.done(start), {"object": "list", "data": models}, "")
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "models", False, 502, self.done(start), None, str(exc))
