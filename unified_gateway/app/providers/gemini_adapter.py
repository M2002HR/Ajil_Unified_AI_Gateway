from __future__ import annotations

from typing import Any, Dict, List

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
from .base import ProviderAdapter, ProviderResult


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                out.append(str(item.get("text", "")))
        return "\n".join([x for x in out if x])
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
    return str(content or "")


def _openai_messages_to_gemini_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role") or "user")
        text = _as_text(m.get("content"))
        if not text.strip():
            continue
        mapped_role = "model" if role == "assistant" else "user"
        contents.append({"role": mapped_role, "parts": [{"text": text}]})
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
        body = {
            "model": model_name,
            "method": "embedContent",
            "content": {"parts": [{"text": str(input_text or "")}]},
        }
        try:
            resp = await self.service.proxy_call(body)
            parsed = resp.json()
            return ProviderResult(
                provider=self.name,
                capability="embeddings",
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
