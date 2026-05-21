from __future__ import annotations

import base64
import json
import time
from typing import Any, AsyncIterator, Dict, List
from uuid import uuid4

from modules.gemini_proxy.api.app.config import (  # type: ignore
    AdminSection,
    AppSection,
    CloudflareSection,
    GeminiSection,
    ProxySection,
    Settings as GeminiSettings,
)
from modules.gemini_proxy.api.app.services import GeminiProxyService  # type: ignore

from ..config import GeminiProviderConfig
from .base import ProviderAdapter, ProviderResult, ProviderStreamResult


def _data_url_to_inline_data(url: str) -> Dict[str, Any] | None:
    value = str(url or "").strip()
    if not value.startswith("data:") or "," not in value:
        return None
    header, encoded = value.split(",", 1)
    if ";base64" not in header:
        return None
    mime_type = header[5:].split(";")[0].strip() or "application/octet-stream"
    try:
        base64.b64decode(encoded, validate=True)
    except Exception:  # noqa: BLE001
        return None
    return {"inlineData": {"mimeType": mime_type, "data": encoded}}


def _openai_content_to_gemini_parts(content: Any) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if isinstance(content, str):
        text = content.strip()
        if text:
            parts.append({"text": text})
        return parts

    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        text = str(content or "").strip()
        if text:
            parts.append({"text": text})
        return parts

    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append({"text": text})
            continue
        if not isinstance(item, dict):
            text = str(item or "").strip()
            if text:
                parts.append({"text": text})
            continue

        item_type = str(item.get("type") or "").strip().lower()
        if item_type in {"text", "input_text"}:
            text = str(item.get("text") or "").strip()
            if text:
                parts.append({"text": text})
            continue

        if item_type in {"image_url", "input_image"}:
            image_value = item.get("image_url")
            if isinstance(image_value, dict):
                image_value = image_value.get("url")
            if not image_value:
                image_value = item.get("url")
            if not image_value:
                continue
            inline_part = _data_url_to_inline_data(str(image_value))
            if inline_part is not None:
                parts.append(inline_part)
            else:
                # Keep non-data URL images as fileData when available.
                parts.append({"fileData": {"fileUri": str(image_value)}})
            continue

        text = str(item.get("text") or "").strip()
        if text:
            parts.append({"text": text})
    return parts


def _openai_messages_to_gemini_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role") or "user")
        parts = _openai_content_to_gemini_parts(m.get("content"))
        if not parts:
            continue
        mapped_role = "model" if role == "assistant" else "user"
        contents.append({"role": mapped_role, "parts": parts})
    return contents or [{"role": "user", "parts": [{"text": ""}]}]


class GeminiAdapter(ProviderAdapter):
    name = "gemini"

    def __init__(self, cfg: GeminiProviderConfig) -> None:
        settings = GeminiSettings(
            app=AppSection(request_timeout_sec=120),
            proxy=ProxySection(
                mode=cfg.mode,
                retry_on_429=cfg.retry_on_429,
                retry_on_5xx=cfg.retry_on_5xx,
                max_retries_per_key=cfg.max_retries_per_key,
                max_retries_on_5xx=cfg.max_retries_on_5xx,
                retry_backoff_sec=cfg.retry_backoff_sec,
                cooloff_sec=cfg.cooloff_sec,
                min_interval_sec=cfg.min_interval_sec,
            ),
            gemini=GeminiSection(
                base_url=cfg.base_url,
                api_version=cfg.api_version,
                default_model=cfg.default_model,
                api_keys=cfg.api_keys,
            ),
            cloudflare=CloudflareSection(
                worker_base_urls=cfg.worker_base_urls,
                worker_route_prefix=cfg.worker_route_prefix,
                auth_token=cfg.worker_auth_token,
                auth_header_name=cfg.worker_auth_header_name,
            ),
            admin=AdminSection(enabled=False),
        )
        self.default_model = cfg.default_model
        self.service = GeminiProxyService(settings)

    async def aclose(self) -> None:
        await self.service.aclose()

    async def chat_completions(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        model_name = str(model or payload.get("model") or self.default_model)
        body: Dict[str, Any] = {"model": model_name}

        if "contents" in payload:
            body["contents"] = payload["contents"]
        else:
            body["contents"] = _openai_messages_to_gemini_contents(list(payload.get("messages") or []))

        generation_config: Dict[str, Any] = {}
        if "temperature" in payload:
            generation_config["temperature"] = payload["temperature"]
        if "max_tokens" in payload:
            generation_config["maxOutputTokens"] = payload["max_tokens"]
        if generation_config:
            body["generationConfig"] = generation_config

        try:
            resp = await self.service.proxy_call(body)
            parsed = resp.json()
            return ProviderResult(
                provider=self.name,
                capability="chat.completions",
                ok=resp.status_code < 400,
                status_code=resp.status_code,
                latency_ms=self.done(start),
                payload=parsed,
                model=model_name,
                headers=dict(resp.headers),
                error="" if resp.status_code < 400 else str(parsed),
            )
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(
                provider=self.name,
                capability="chat.completions",
                ok=False,
                status_code=502,
                latency_ms=self.done(start),
                payload=None,
                model=model_name,
                error=str(exc),
            )

    async def chat_completions_stream(
        self,
        payload: Dict[str, Any],
        model: str | None = None,
        *,
        timeout_sec: float | None = None,
    ) -> ProviderStreamResult:
        start = self.started()
        model_name = str(model or payload.get("model") or self.default_model)
        body: Dict[str, Any] = {"model": model_name}
        if "contents" in payload:
            body["contents"] = payload["contents"]
        else:
            body["contents"] = _openai_messages_to_gemini_contents(list(payload.get("messages") or []))

        generation_config: Dict[str, Any] = {}
        if "temperature" in payload:
            generation_config["temperature"] = payload["temperature"]
        if "max_tokens" in payload:
            generation_config["maxOutputTokens"] = payload["max_tokens"]
        if generation_config:
            body["generationConfig"] = generation_config

        api_version = str(self.service.settings.gemini.api_version)
        method = "streamGenerateContent"
        params = {"alt": "sse"}

        retries = max(0, int(self.service.settings.proxy.max_retries_per_key) - 1)
        for _ in range(retries + 1):
            await self.service.limiter.wait()
            active_worker_slot = None
            active_worker_url = None
            active_key_slot = None
            active_key_mask = None
            if self.service.settings.proxy.mode == "cloudflare_worker":
                if not self.service.worker_rr:
                    return ProviderStreamResult(
                        provider=self.name,
                        capability="chat.completions",
                        ok=False,
                        status_code=503,
                        latency_ms=self.done(start),
                        model=model_name,
                        error="No Cloudflare worker URL configured",
                    )
                active_worker_slot = self.service.worker_rr.index + 1
                active_worker_url = self.service.worker_rr.active
                url = self.service._build_worker_url(api_version=api_version, model=model_name, method=method)
            else:
                if not self.service.gemini_key_rr:
                    return ProviderStreamResult(
                        provider=self.name,
                        capability="chat.completions",
                        ok=False,
                        status_code=503,
                        latency_ms=self.done(start),
                        model=model_name,
                        error="No Gemini API key configured",
                    )
                active_key_slot = self.service.gemini_key_rr.index + 1
                active_key_mask = self.service._mask_key(self.service.gemini_key_rr.active)
                url = self.service._build_direct_gemini_url(api_version=api_version, model=model_name, method=method)

            headers = self.service._build_headers(request_id=None)
            headers["accept"] = "text/event-stream"
            request = self.service.client.build_request("POST", url, headers=headers, json={k: v for k, v in body.items() if k != "model"}, params=params)
            try:
                # httpx.AsyncClient.send() in 0.28.x does not accept a timeout kwarg.
                # Per-attempt timeout is enforced by router asyncio.wait_for() wrapping this call.
                resp = await self.service.client.send(request, stream=True)
            except Exception as exc:  # noqa: BLE001
                if self.service.settings.proxy.mode == "cloudflare_worker":
                    if self.service.worker_rr:
                        self.service.worker_rr.next()
                elif self.service.gemini_key_rr:
                    self.service.gemini_key_rr.next()
                return ProviderStreamResult(
                    provider=self.name,
                    capability="chat.completions",
                    ok=False,
                    status_code=502,
                    latency_ms=self.done(start),
                    model=model_name,
                    error=str(exc),
                )

            self.service._set_proxy_metadata_headers(
                resp,
                worker_slot=active_worker_slot,
                worker_url=active_worker_url,
                key_slot=active_key_slot,
                key_pool_size=(len(self.service.gemini_key_rr.values) if self.service.gemini_key_rr else None),
                attempts=1,
                key_rotated=False,
            )
            if active_key_mask:
                resp.headers["x-proxy-key-mask"] = active_key_mask

            if resp.status_code < 400:
                chunk_id = f"chatcmpl-{uuid4().hex[:12]}"
                created_ts = int(time.time())

                async def _stream() -> AsyncIterator[bytes]:
                    emitted_role = False
                    try:
                        async for line in resp.aiter_lines():
                            line = str(line or "").strip()
                            if not line or not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if not raw:
                                continue
                            if raw == "[DONE]":
                                done_chunk = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": model_name,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                }
                                yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                                yield b"data: [DONE]\n\n"
                                return
                            try:
                                event_payload = json.loads(raw)
                            except Exception:
                                continue
                            candidates = list(event_payload.get("candidates") or [])
                            text_chunks: List[str] = []
                            finish_reason = None
                            if candidates:
                                first = candidates[0] or {}
                                finish_reason = first.get("finishReason")
                                parts = ((first.get("content") or {}).get("parts") or [])
                                for part in parts:
                                    if isinstance(part, dict):
                                        txt = str(part.get("text") or "")
                                        if txt:
                                            text_chunks.append(txt)
                            delta_payload: Dict[str, Any] = {}
                            if not emitted_role:
                                delta_payload["role"] = "assistant"
                                emitted_role = True
                            text_out = "".join(text_chunks)
                            if text_out:
                                delta_payload["content"] = text_out
                            chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta_payload,
                                        "finish_reason": "stop" if finish_reason else None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                        terminal = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(terminal, ensure_ascii=False)}\n\n".encode("utf-8")
                        yield b"data: [DONE]\n\n"
                    finally:
                        await resp.aclose()

                return ProviderStreamResult(
                    provider=self.name,
                    capability="chat.completions",
                    ok=True,
                    status_code=200,
                    latency_ms=self.done(start),
                    model=model_name,
                    headers=dict(resp.headers),
                    stream=_stream(),
                )

            error_text = ""
            try:
                raw_error = await resp.aread()
                error_text = raw_error.decode("utf-8", errors="ignore")
            finally:
                await resp.aclose()

            should_retry = (
                self.service.settings.proxy.retry_on_429 and self.service._is_429(resp)
            ) or (
                bool(self.service.settings.proxy.retry_on_5xx)
                and resp.status_code >= 500
            )
            if should_retry:
                if self.service.settings.proxy.mode == "cloudflare_worker":
                    if self.service.worker_rr:
                        self.service.worker_rr.next()
                elif self.service.gemini_key_rr:
                    self.service.gemini_key_rr.next()
                continue
            return ProviderStreamResult(
                provider=self.name,
                capability="chat.completions",
                ok=False,
                status_code=int(resp.status_code),
                latency_ms=self.done(start),
                model=model_name,
                headers=dict(resp.headers),
                error=error_text or f"upstream status={resp.status_code}",
            )

        return ProviderStreamResult(
            provider=self.name,
            capability="chat.completions",
            ok=False,
            status_code=429,
            latency_ms=self.done(start),
            model=model_name,
            error="all stream retries exhausted",
        )

    async def responses(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        # Gemini does not expose an OpenAI Responses endpoint; map to generateContent.
        mapped = dict(payload)
        if "input" in mapped and "messages" not in mapped and "contents" not in mapped:
            mapped["messages"] = [{"role": "user", "content": str(mapped["input"])}]
        return await self.chat_completions(mapped, model=model)

    async def embeddings(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        model_name = str(model or payload.get("model") or "text-embedding-004")
        input_text = payload.get("input")
        if isinstance(input_text, list):
            input_text = "\n".join([str(x) for x in input_text])
        base_body = {
            "method": "embedContent",
            "content": {"parts": [{"text": str(input_text or "")}]},
        }
        fallback_models = [model_name, "gemini-embedding-001", "gemini-embedding-2", "gemini-embedding-2-preview"]
        tried_models: list[str] = []

        try:
            for candidate_model in fallback_models:
                if candidate_model in tried_models:
                    continue
                tried_models.append(candidate_model)
                body = dict(base_body)
                body["model"] = candidate_model
                resp = await self.service.proxy_call(body)
                parsed = resp.json()
                if resp.status_code < 400:
                    return ProviderResult(
                        provider=self.name,
                        capability="embeddings",
                        ok=True,
                        status_code=resp.status_code,
                        latency_ms=self.done(start),
                        payload=parsed,
                        model=candidate_model,
                        headers=dict(resp.headers),
                        error="",
                    )
                error_message = str((parsed.get("error") or {}).get("message") or "").lower() if isinstance(parsed, dict) else ""
                can_fallback = "not supported for embedcontent" in error_message or "is not found for api version" in error_message
                if not can_fallback:
                    return ProviderResult(
                        provider=self.name,
                        capability="embeddings",
                        ok=False,
                        status_code=resp.status_code,
                        latency_ms=self.done(start),
                        payload=parsed,
                        model=candidate_model,
                        headers=dict(resp.headers),
                        error=str(parsed),
                    )

            # exhausted all fallback embedding models
            return ProviderResult(
                provider=self.name,
                capability="embeddings",
                ok=False,
                status_code=404,
                latency_ms=self.done(start),
                payload={"error": {"message": "No supported Gemini embedding model available"}},
                model=(tried_models[-1] if tried_models else model_name),
                error="No supported Gemini embedding model available",
            )
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(
                provider=self.name,
                capability="embeddings",
                ok=False,
                status_code=502,
                latency_ms=self.done(start),
                payload=None,
                model=model_name,
                error=str(exc),
            )

    async def list_models(self) -> ProviderResult:
        start = self.started()
        try:
            payload = await self.service.get_models(capability="all", include_preview=True)
            return ProviderResult(
                provider=self.name,
                capability="models",
                ok=True,
                status_code=200,
                latency_ms=self.done(start),
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(
                provider=self.name,
                capability="models",
                ok=False,
                status_code=502,
                latency_ms=self.done(start),
                payload=None,
                error=str(exc),
            )


def gemini_to_openai_chat(result: ProviderResult) -> Dict[str, Any]:
    payload = result.payload or {}
    candidates = payload.get("candidates") or []
    text_out = ""
    if candidates:
        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        chunks = [str(p.get("text", "")) for p in parts if isinstance(p, dict)]
        text_out = "\n".join([c for c in chunks if c]).strip()

    return {
        "id": payload.get("responseId", "gemini-proxy-response"),
        "object": "chat.completion",
        "created": None,
        "model": result.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text_out}, "finish_reason": "stop"}],
        "usage": payload.get("usageMetadata", {}),
        "proxy_metadata": {
            "provider": "gemini",
            "status_code": result.status_code,
            "latency_ms": round(result.latency_ms, 2),
        },
    }
