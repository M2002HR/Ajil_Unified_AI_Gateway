from __future__ import annotations

import json
from typing import Any, Dict

from modules.groq_proxy.api.app.config import (  # type: ignore
    AdminSection,
    AppSection,
    GroqSection,
    ProxySection,
    Settings as GroqSettings,
)
from modules.groq_proxy.api.app.services import GroqProxyService  # type: ignore

from ..config import GroqProviderConfig
from .base import ProviderAdapter, ProviderResult


class GroqAdapter(ProviderAdapter):
    name = "groq"

    def __init__(self, cfg: GroqProviderConfig) -> None:
        keys = list(cfg.api_keys)
        if not keys:
            keys = ["__missing__"]

        settings = GroqSettings(
            app=AppSection(request_timeout_sec=90),
            proxy=ProxySection(
                retry_on_429=cfg.retry_on_429,
                retry_on_5xx=cfg.retry_on_5xx,
                max_retries_per_key=cfg.max_retries_per_key,
                max_retries_on_5xx=cfg.max_retries_on_5xx,
                retry_backoff_sec=cfg.retry_backoff_sec,
                cooloff_sec=cfg.cooloff_sec,
                min_interval_sec=cfg.min_interval_sec,
            ),
            groq=GroqSection(
                base_url=cfg.base_url,
                api_key=keys[0],
                api_keys=keys,
                stt_primary_model=cfg.stt_primary_model,
                stt_fallback_model=cfg.stt_fallback_model,
                stt_language=cfg.stt_language,
                stt_temperature=cfg.stt_temperature,
                stt_response_format=cfg.stt_response_format,
                stt_prompt=cfg.stt_prompt,
                tts_default_model=cfg.tts_default_model,
                tts_default_voice=cfg.tts_default_voice,
                tts_default_response_format=cfg.tts_default_response_format,
            ),
            admin=AdminSection(enabled=False),
        )
        self.service = GroqProxyService(settings)

    async def aclose(self) -> None:
        await self.service.aclose()

    @staticmethod
    def _parse(resp: Any) -> Any:
        ctype = str(resp.headers.get("content-type", "application/json")).lower()
        if "application/json" in ctype:
            try:
                return resp.json()
            except json.JSONDecodeError:
                return {"raw": resp.text}
        return resp.content

    async def chat_completions(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model
        try:
            resp = await self.service.forward_json(path="/chat/completions", payload=body)
            parsed = self._parse(resp)
            return ProviderResult(self.name, "chat.completions", resp.status_code < 400, resp.status_code, self.done(start), parsed, "" if resp.status_code < 400 else str(parsed), str(body.get("model", "")), dict(resp.headers))
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "chat.completions", False, 502, self.done(start), None, str(exc), str(body.get("model", "")))

    async def responses(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model
        try:
            resp = await self.service.forward_json(path="/responses", payload=body)
            parsed = self._parse(resp)
            return ProviderResult(self.name, "responses", resp.status_code < 400, resp.status_code, self.done(start), parsed, "" if resp.status_code < 400 else str(parsed), str(body.get("model", "")), dict(resp.headers))
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "responses", False, 502, self.done(start), None, str(exc), str(body.get("model", "")))

    async def embeddings(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model
        try:
            resp = await self.service.forward_json(path="/embeddings", payload=body)
            parsed = self._parse(resp)
            return ProviderResult(self.name, "embeddings", resp.status_code < 400, resp.status_code, self.done(start), parsed, "" if resp.status_code < 400 else str(parsed), str(body.get("model", "")), dict(resp.headers))
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "embeddings", False, 502, self.done(start), None, str(exc), str(body.get("model", "")))

    async def tts(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model
        try:
            resp = await self.service.forward_json(path="/audio/speech", payload=body)
            parsed = self._parse(resp)
            return ProviderResult(self.name, "audio.speech", resp.status_code < 400, resp.status_code, self.done(start), parsed, "" if resp.status_code < 400 else str(parsed), str(body.get("model", "")), dict(resp.headers))
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "audio.speech", False, 502, self.done(start), None, str(exc), str(body.get("model", "")))

    async def stt(self, *, audio_bytes: bytes, filename: str, content_type: str, language: str | None) -> ProviderResult:
        start = self.started()
        try:
            parsed = await self.service.transcribe(
                audio_bytes=audio_bytes,
                filename=filename,
                content_type=content_type,
                language=language,
                prompt=None,
                response_format=None,
                temperature=None,
            )
            model_used = ""
            if isinstance(parsed, dict):
                model_used = str(parsed.get("model_used") or "")
            return ProviderResult(self.name, "audio.transcriptions", True, 200, self.done(start), parsed, "", model_used)
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "audio.transcriptions", False, 502, self.done(start), None, str(exc))

    async def list_models(self) -> ProviderResult:
        start = self.started()
        try:
            resp = await self.service.forward_models(path="/models")
            parsed = self._parse(resp)
            return ProviderResult(self.name, "models", resp.status_code < 400, resp.status_code, self.done(start), parsed, "" if resp.status_code < 400 else str(parsed), headers=dict(resp.headers))
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "models", False, 502, self.done(start), None, str(exc))
