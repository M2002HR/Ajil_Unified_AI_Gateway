from __future__ import annotations

import asyncio
import base64
import time
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
    _PLACEHOLDER_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Zx3sAAAAASUVORK5CYII="

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
        self.default_image_model = cfg.default_image_model
        self.local_placeholder_on_failure = bool(cfg.local_placeholder_on_failure)

    async def aclose(self) -> None:
        await self.service.close()

    async def image_generations(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model

        def _placeholder_result(model_name: str, reason: str, headers: Dict[str, str] | None = None) -> ProviderResult:
            placeholder_payload = {
                "created": int(time.time()),
                "data": [
                    {
                        "b64_json": self._PLACEHOLDER_PNG_B64,
                        "url": f"data:image/png;base64,{self._PLACEHOLDER_PNG_B64}",
                        "revised_prompt": "",
                    }
                ],
                "provider_fallback": {
                    "reason": reason,
                    "route": "local-placeholder",
                    "model": model_name,
                    "response_format_requested": str(body.get("response_format") or "b64_json"),
                },
            }
            merged_headers = {**dict(headers or {}), "x-pollinations-fallback": "local-placeholder"}
            return ProviderResult(
                self.name,
                "images.generations",
                True,
                200,
                self.done(start),
                placeholder_payload,
                "",
                model_name,
                merged_headers,
            )

        try:
            status, parsed, headers = await self.service.generate_image(body)
            if status < 400:
                return ProviderResult(self.name, "images.generations", True, status, self.done(start), parsed, "", str(body.get("model", "")), headers)

            error_payload = parsed if isinstance(parsed, dict) else {}
            error_text = str((error_payload.get("error") or {}).get("message") or parsed)
            error_code = str((error_payload.get("error") or {}).get("code") or "")
            status_i = int(status)
            low_error_text = error_text.lower()
            needs_free_fallback = (
                status_i in {402, 408, 409, 425, 429, 500, 502, 503, 504}
                or "payment_required" in error_code.lower()
                or "insufficient balance" in low_error_text
                or "timeout" in low_error_text
                or "timed out" in low_error_text
                or "connection" in low_error_text
                or "upstream" in low_error_text
            )

            if not needs_free_fallback:
                return ProviderResult(
                    self.name,
                    "images.generations",
                    False,
                    status,
                    self.done(start),
                    parsed,
                    str(parsed),
                    str(body.get("model", "")),
                    headers,
                )

            # Fallback path: use free public image endpoint with a free model.
            try:
                free_models = await self.service.list_free_image_models(force_refresh=False)
            except Exception:
                free_models = []
            free_names = [str((m or {}).get("name") or "").strip() for m in free_models]
            free_names = [name for name in free_names if name]

            requested_model = str(body.get("model") or "")
            preferred: list[str] = []
            if requested_model and requested_model in free_names:
                preferred.append(requested_model)
            if self.default_image_model in free_names and self.default_image_model not in preferred:
                preferred.append(self.default_image_model)
            if "flux" in free_names and "flux" not in preferred:
                preferred.append("flux")
            for item in free_names:
                if item not in preferred:
                    preferred.append(item)
            if not preferred:
                preferred = ["flux"]

            fallback_body = dict(body)
            fallback_body.setdefault("size", "512x512")
            fallback_body.setdefault("quality", "low")
            chosen_model = preferred[0]
            # First fallback path: still use authenticated OpenAI-compatible endpoint but with free model.
            free_attempt_error = ""
            for candidate_model in preferred:
                fallback_body["model"] = candidate_model
                free_status, free_parsed, free_headers = await self.service.generate_image(fallback_body)
                if free_status < 400:
                    merged_headers = {**headers, **free_headers, "x-pollinations-fallback": "free-model"}
                    if isinstance(free_parsed, dict):
                        free_parsed.setdefault(
                            "provider_fallback",
                            {
                                "reason": "paid-quota-exhausted",
                                "route": "free-model",
                                "model": candidate_model,
                            },
                        )
                    return ProviderResult(
                        self.name,
                        "images.generations",
                        True,
                        free_status,
                        self.done(start),
                        free_parsed,
                        "",
                        candidate_model,
                        merged_headers,
                    )
                chosen_model = candidate_model
                free_attempt_error = str(free_parsed)

            prompt = str(body.get("prompt") or "")
            size = str(body.get("size") or "1024x1024")
            width = None
            height = None
            if "x" in size:
                left, right = size.split("x", 1)
                try:
                    width = int(left.strip())
                    height = int(right.strip())
                except ValueError:
                    width = None
                    height = None

            get_status, get_payload, get_headers, out_type = await self.service.generate_image_get_public(
                prompt=prompt,
                params={
                    "model": chosen_model,
                    "width": width,
                    "height": height,
                    "seed": body.get("seed"),
                    "safe": body.get("safe"),
                    "quality": body.get("quality"),
                    "enhance": body.get("enhance"),
                    "transparent": body.get("transparent"),
                },
            )

            if get_status >= 400:
                if self.local_placeholder_on_failure:
                    return _placeholder_result(chosen_model, "upstream-unavailable", headers={**headers, **get_headers})
                merged_headers = {**headers, **get_headers, "x-pollinations-fallback": "public-image-failed"}
                return ProviderResult(
                    self.name,
                    "images.generations",
                    False,
                    get_status,
                    self.done(start),
                    {
                        "free_attempt_error": free_attempt_error,
                        "public_attempt_error": get_payload,
                    },
                    str(get_payload),
                    chosen_model,
                    merged_headers,
                )

            b64_data = ""
            if isinstance(get_payload, (bytes, bytearray)):
                b64_data = base64.b64encode(bytes(get_payload)).decode("ascii")

            requested_format = str(body.get("response_format") or "b64_json")
            # Public image route returns binary; expose b64_json and a data URL for compatibility.
            mime = str(out_type or "image/jpeg")
            fallback_payload = {
                "created": int(time.time()),
                "data": [
                    {
                        "b64_json": b64_data,
                        "url": f"data:{mime};base64,{b64_data}" if b64_data else "",
                        "revised_prompt": "",
                    }
                ],
                "provider_fallback": {
                    "reason": "paid-quota-exhausted",
                    "route": "public-image",
                    "model": chosen_model,
                    "response_format_requested": requested_format,
                },
            }
            merged_headers = {**headers, **get_headers, "x-pollinations-fallback": "public-image"}
            return ProviderResult(
                self.name,
                "images.generations",
                True,
                200,
                self.done(start),
                fallback_payload,
                "",
                chosen_model,
                merged_headers,
            )
        except Exception as exc:  # noqa: BLE001
            if self.local_placeholder_on_failure:
                return _placeholder_result(str(body.get("model", "") or self.default_image_model), f"adapter-exception:{exc}")
            return ProviderResult(self.name, "images.generations", False, 502, self.done(start), None, str(exc), str(body.get("model", "")))

    async def list_models(self) -> ProviderResult:
        start = self.started()
        try:
            image_models, free_models, v1_payload = await asyncio.gather(
                self.service.list_image_models(force_refresh=False),
                self.service.list_free_image_models(force_refresh=False),
                self.service.list_v1_models(force_refresh=False),
            )
            free_set = {str((m or {}).get("name") or "") for m in free_models}
            out: list[dict[str, Any]] = []
            for model_row in image_models:
                row = dict(model_row or {})
                model_name = str(row.get("name") or "")
                row["is_free"] = model_name in free_set
                row["provider"] = "pollinations"
                out.append(row)

            return ProviderResult(
                self.name,
                "models",
                True,
                200,
                self.done(start),
                {
                    "object": "list",
                    "provider": "pollinations",
                    "count": len(out),
                    "free_count": len(free_models),
                    "data": out,
                    "v1_models": v1_payload,
                },
                "",
            )
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "models", False, 502, self.done(start), None, str(exc))
