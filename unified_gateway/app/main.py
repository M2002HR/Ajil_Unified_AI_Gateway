from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .config import Settings, get_settings
from .model_catalog import filter_models as filter_catalog_models
from .model_catalog import normalize_models as normalize_catalog_models
from .model_catalog import summarize_models as summarize_catalog_models
from .observability import get_request_id, log_event, now_iso, reset_request_id, set_request_id, setup_logging
from .providers.base import ProviderResult
from .providers.gemini_adapter import gemini_to_openai_chat
from .providers.registry import ProviderRegistry
from .router.engine import RoutingEngine
from .schemas import HealthResponse, OrchestrateRequest, RouterOptions
from .state.event_tracker import EventTracker
from .state.rate_limit import RateLimitGuard
from .state.usage_tracker import UsageTracker


class AppState:
    def __init__(
        self,
        settings: Settings,
        registry: ProviderRegistry,
        guard: RateLimitGuard,
        router: RoutingEngine,
        usage_tracker: UsageTracker,
        event_tracker: EventTracker,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.guard = guard
        self.router = router
        self.usage_tracker = usage_tracker
        self.event_tracker = event_tracker
        self.models_catalog_cache: Dict[str, Any] = {
            "fetched_at": "",
            "ttl_sec": 180,
            "provider_rows": [],
            "items": [],
        }
        self.models_catalog_lock = asyncio.Lock()
        self.image_options_cache: Dict[str, Any] = {}
        self.image_options_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(level=settings.app.log_level, use_json=settings.logging.json_logs, enabled=settings.logging.enabled)
    logger = logging.getLogger("uag.app")

    if settings.proxy.enabled and settings.proxy.url.strip():
        proxy_url = settings.proxy.url.strip()
        # Shared outbound proxy for providers that rely on env-based httpx proxy discovery.
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["ALL_PROXY"] = proxy_url
        # Gemini service also checks explicit per-service proxy envs.
        os.environ["GEMINI_HTTP_PROXY"] = proxy_url
        os.environ["GEMINI_HTTPS_PROXY"] = proxy_url

    registry = ProviderRegistry.build(settings)
    guard = RateLimitGuard(
        redis_url=settings.redis.url,
        key_prefix=settings.redis.key_prefix,
        required=settings.redis.required,
    )

    if settings.redis.required:
        ok = await guard.ping()
        if not ok:
            raise RuntimeError("Redis is required but not reachable")

    usage_tracker = UsageTracker(max_events=settings.usage.max_events)
    event_tracker = EventTracker(max_events=settings.logging.max_events)
    router = RoutingEngine(settings, registry, guard, usage_tracker=usage_tracker, event_tracker=event_tracker)
    app.state.ctx = AppState(
        settings=settings,
        registry=registry,
        guard=guard,
        router=router,
        usage_tracker=usage_tracker,
        event_tracker=event_tracker,
    )
    log_event(
        logger,
        event="app.startup",
        level=logging.INFO,
        message="application started",
        providers=registry.names(),
        redis_required=settings.redis.required,
        proxy_enabled=settings.proxy.enabled,
        log_json=settings.logging.json_logs,
    )
    await event_tracker.record(
        event_type="app.startup",
        level="INFO",
        message="application started",
        data={
            "providers": registry.names(),
            "redis_required": bool(settings.redis.required),
            "proxy_enabled": bool(settings.proxy.enabled),
            "log_json": bool(settings.logging.json_logs),
        },
    )
    try:
        yield
    finally:
        log_event(logger, event="app.shutdown", level=logging.INFO, message="application shutting down")
        await event_tracker.record(event_type="app.shutdown", level="INFO", message="application shutting down", data={})
        await guard.close()
        await registry.close()


settings = get_settings()
app = FastAPI(
    title="Unified AI Gateway",
    version="1.0.0",
    docs_url="/docs" if settings.app.docs_enabled else None,
    redoc_url="/redoc" if settings.app.docs_enabled else None,
    openapi_url="/openapi.json" if settings.app.docs_enabled else None,
    lifespan=lifespan,
)
app_logger = logging.getLogger("uag.http")
ADMIN_UI_ROOT = Path(__file__).resolve().parent / "admin_ui"
STUDIO_UI_ROOT = Path(__file__).resolve().parent / "studio_ui"


def _ctx(request: Request) -> AppState:
    return request.app.state.ctx


def _to_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        out = int(str(value))
    except (TypeError, ValueError):
        out = default
    return max(min_value, min(max_value, out))


def _router_failure_status(ctx: AppState, result: Dict[str, Any], default: int = 502) -> int:
    code = result.get("status_code")
    try:
        parsed = int(code)
    except (TypeError, ValueError):
        parsed = default

    return max(400, min(599, parsed))


def _family_from_model_id(model_id: str) -> str:
    lid = str(model_id or "").lower()
    for name in ("gemma", "gemini", "llama", "mixtral", "qwen", "gpt", "whisper", "flux", "seedream", "kontext", "orpheus"):
        if name in lid:
            return name
    return "other"


def _build_static_fallback_catalog(settings: Settings) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_row(
        provider: str,
        model_id: str,
        *,
        capabilities: List[str],
        model_type: str,
        input_modalities: List[str],
        output_modalities: List[str],
        label: str | None = None,
        preview: bool = False,
    ) -> None:
        normalized_id = str(model_id or "").strip()
        if not normalized_id:
            return
        key = (provider, normalized_id)
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "provider": provider,
                "id": normalized_id,
                "name": normalized_id,
                "label": str(label or normalized_id),
                "family": _family_from_model_id(normalized_id),
                "capabilities": list(capabilities),
                "model_type": model_type,
                "input_modalities": list(input_modalities),
                "output_modalities": list(output_modalities),
                "preview": bool(preview),
                "paid_only": None,
                "context_window": None,
                "max_output_tokens": None,
                "raw": {"source": "settings_fallback"},
            }
        )

    # Gemini
    add_row(
        "gemini",
        settings.gemini.default_model,
        capabilities=["chat.completions", "responses"],
        model_type="llm",
        input_modalities=["text"],
        output_modalities=["text"],
    )
    add_row(
        "gemini",
        "gemini-embedding-001",
        capabilities=["embeddings"],
        model_type="embedding",
        input_modalities=["text"],
        output_modalities=["embedding"],
    )
    add_row(
        "gemini",
        "gemini-embedding-2",
        capabilities=["embeddings"],
        model_type="embedding",
        input_modalities=["text"],
        output_modalities=["embedding"],
        preview=True,
    )

    # Groq
    add_row(
        "groq",
        "llama-3.3-70b-versatile",
        capabilities=["chat.completions", "responses"],
        model_type="llm",
        input_modalities=["text"],
        output_modalities=["text"],
    )
    add_row(
        "groq",
        settings.groq.stt_primary_model,
        capabilities=["audio.transcriptions"],
        model_type="audio_stt",
        input_modalities=["audio"],
        output_modalities=["text"],
    )
    add_row(
        "groq",
        settings.groq.stt_fallback_model,
        capabilities=["audio.transcriptions"],
        model_type="audio_stt",
        input_modalities=["audio"],
        output_modalities=["text"],
    )
    add_row(
        "groq",
        settings.groq.tts_default_model,
        capabilities=["audio.speech"],
        model_type="audio_tts",
        input_modalities=["text"],
        output_modalities=["audio"],
    )

    # Pollinations
    add_row(
        "pollinations",
        settings.pollinations.default_image_model,
        capabilities=["images.generations"],
        model_type="image",
        input_modalities=["text"],
        output_modalities=["image"],
    )
    add_row(
        "pollinations",
        "flux",
        capabilities=["images.generations"],
        model_type="image",
        input_modalities=["text"],
        output_modalities=["image"],
    )

    return rows


async def _refresh_models_catalog(
    ctx: AppState,
    *,
    refresh: bool,
    timeout_sec: float = 30.0,
) -> Dict[str, Any]:
    now_ts = time.monotonic()
    cache = ctx.models_catalog_cache
    ttl_sec = int(cache.get("ttl_sec") or 180)
    fetched_at = str(cache.get("fetched_at") or "")
    cache_age_ok = False
    if fetched_at:
        try:
            cache_age_ok = (now_ts - float(cache.get("fetched_monotonic", 0.0))) <= max(10, ttl_sec)
        except Exception:  # noqa: BLE001
            cache_age_ok = False

    if not refresh and cache_age_ok and cache.get("items") is not None:
        return {
            "generated_at": now_iso(),
            "from_cache": True,
            "fetched_at": cache.get("fetched_at"),
            "provider_rows": list(cache.get("provider_rows") or []),
            "items": list(cache.get("items") or []),
            "fallback_applied": bool(cache.get("fallback_applied")),
        }

    async with ctx.models_catalog_lock:
        # Double-check after locking to avoid duplicate refresh.
        now_ts = time.monotonic()
        cache = ctx.models_catalog_cache
        if not refresh and cache.get("items") is not None:
            try:
                if (now_ts - float(cache.get("fetched_monotonic", 0.0))) <= max(10, int(cache.get("ttl_sec") or 180)):
                    return {
                        "generated_at": now_iso(),
                        "from_cache": True,
                        "fetched_at": cache.get("fetched_at"),
                        "provider_rows": list(cache.get("provider_rows") or []),
                        "items": list(cache.get("items") or []),
                        "fallback_applied": bool(cache.get("fallback_applied")),
                    }
            except Exception:  # noqa: BLE001
                pass

        async def fetch_provider(provider: str, adapter: Any) -> Dict[str, Any]:
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(adapter.list_models(), timeout=max(2.0, timeout_sec))
                payload = result.payload if result.ok else {}
                normalized = normalize_catalog_models(provider, payload if isinstance(payload, dict) else {})
                return {
                    "provider": provider,
                    "ok": bool(result.ok),
                    "status_code": int(result.status_code),
                    "latency_ms": round(float(result.latency_ms), 3),
                    "error": str(result.error or ""),
                    "count": len(normalized),
                    "items": normalized,
                }
            except asyncio.TimeoutError:
                return {
                    "provider": provider,
                    "ok": False,
                    "status_code": 504,
                    "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
                    "error": f"list_models timeout after {timeout_sec}s",
                    "count": 0,
                    "items": [],
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "provider": provider,
                    "ok": False,
                    "status_code": 502,
                    "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
                    "error": str(exc),
                    "count": 0,
                    "items": [],
                }

        jobs = [fetch_provider(name, adapter) for name, adapter in ctx.registry.providers.items() if hasattr(adapter, "list_models")]
        provider_rows = await asyncio.gather(*jobs)
        all_items: List[Dict[str, Any]] = []
        for row in provider_rows:
            all_items.extend(list(row.get("items") or []))

        fallback_applied = False
        if not all_items:
            fallback_applied = True
            all_items = _build_static_fallback_catalog(ctx.settings)
            counts_by_provider: Dict[str, int] = {}
            for item in all_items:
                p = str(item.get("provider") or "")
                counts_by_provider[p] = counts_by_provider.get(p, 0) + 1
            for row in provider_rows:
                provider_key = str(row.get("provider") or "")
                row["fallback_count"] = counts_by_provider.get(provider_key, 0)

        ctx.models_catalog_cache = {
            "fetched_at": now_iso(),
            "fetched_monotonic": time.monotonic(),
            "ttl_sec": 180,
            "provider_rows": provider_rows,
            "items": all_items,
            "fallback_applied": fallback_applied,
        }
        return {
            "generated_at": now_iso(),
            "from_cache": False,
            "fetched_at": ctx.models_catalog_cache["fetched_at"],
            "provider_rows": provider_rows,
            "items": all_items,
            "fallback_applied": fallback_applied,
        }


def _normalize_image_option_values(values: Any) -> List[str]:
    if isinstance(values, str):
        parts = [x.strip() for x in values.split(",") if x.strip()]
        return list(dict.fromkeys(parts))
    if isinstance(values, list):
        out = [str(x).strip() for x in values if str(x).strip()]
        return list(dict.fromkeys(out))
    return []


def _extract_allowed_values_from_error(message: str, field_name: str) -> List[str]:
    text = str(message or "")
    patterns = [
        rf"{re.escape(field_name)}\s*(?:must be|should be|is)\s*(?:one of|in)\s*\[([^\]]+)\]",
        rf"{re.escape(field_name)}\s*(?:allowed values|valid values?)\s*[:=]\s*([A-Za-z0-9_,\-\sx]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1)
        candidates = [x.strip().strip("'\"") for x in re.split(r"[,\s]+", raw) if x.strip().strip("'\"")]
        if candidates:
            return list(dict.fromkeys(candidates))
    return []


def _extract_image_options_from_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    size_keys = ["sizes", "supported_sizes", "allowed_sizes", "size_options", "resolutions"]
    quality_keys = ["qualities", "supported_qualities", "allowed_qualities", "quality_options"]
    defaults = {
        "size": str(raw.get("default_size") or raw.get("size") or "").strip(),
        "quality": str(raw.get("default_quality") or raw.get("quality") or "").strip(),
    }

    sizes: List[str] = []
    qualities: List[str] = []
    for key in size_keys:
        sizes.extend(_normalize_image_option_values(raw.get(key)))
    for key in quality_keys:
        qualities.extend(_normalize_image_option_values(raw.get(key)))

    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    if isinstance(params, dict):
        sizes.extend(_normalize_image_option_values(params.get("size")))
        qualities.extend(_normalize_image_option_values(params.get("quality")))

    return {
        "sizes": list(dict.fromkeys([x for x in sizes if x])),
        "qualities": list(dict.fromkeys([x for x in qualities if x])),
        "default_size": defaults["size"],
        "default_quality": defaults["quality"],
    }


async def _resolve_image_options(
    ctx: AppState,
    *,
    provider: str,
    model: str,
    refresh: bool = False,
) -> Dict[str, Any]:
    provider_key = str(provider or "").strip().lower()
    model_name = str(model or "").strip()
    if not provider_key or not model_name:
        raise HTTPException(status_code=400, detail="provider and model are required")

    cache_key = f"{provider_key}:{model_name}"
    ttl_sec = 1800
    now_ts = time.monotonic()

    if not refresh:
        async with ctx.image_options_lock:
            cached = ctx.image_options_cache.get(cache_key)
            if cached and (now_ts - float(cached.get("fetched_monotonic", 0.0))) <= ttl_sec:
                return dict(cached.get("payload") or {})

    catalog = await _refresh_models_catalog(ctx, refresh=False)
    row = next(
        (
            item
            for item in (catalog.get("items") or [])
            if str(item.get("provider") or "").lower() == provider_key and str(item.get("id") or "") == model_name
        ),
        None,
    )
    raw = dict((row or {}).get("raw") or {})
    extracted = _extract_image_options_from_raw(raw)

    size_candidates = extracted["sizes"] or ["512x512", "768x768", "1024x1024", "1024x1536", "1536x1024"]
    quality_candidates = extracted["qualities"] or ["low", "medium", "high", "standard", "hd", "auto"]
    default_size = extracted["default_size"] or ("1024x1024" if "1024x1024" in size_candidates else size_candidates[0])
    default_quality = extracted["default_quality"] or ("medium" if "medium" in quality_candidates else quality_candidates[0])

    async def try_once(size: str, quality: str, timeout_sec: float = 8.0) -> Dict[str, Any]:
        payload = {
            "model": [{"provider": provider_key, "model": model_name, "priority": 0}],
            "prompt": "Minimal test image.",
            "size": size,
            "quality": quality,
            "n": 1,
            "response_format": "b64_json",
        }
        options = RouterOptions(
            providers=[provider_key],
            strategy="fallback_chain",
            mode="limit_safe",
            timeout_sec=timeout_sec,
            max_attempts=1,
        )
        result = await ctx.router.dispatch("images.generations", payload, options)
        if result.get("ok"):
            return {"ok": True, "status_code": 200, "error": ""}
        first_error = ""
        for item in list(result.get("results") or []):
            err = str(item.get("error") or "")
            if err:
                first_error = err
                break
        return {
            "ok": False,
            "status_code": int(result.get("status_code") or 502),
            "error": first_error or str(result.get("error_type") or "probe_failed"),
        }

    detected_qualities: List[str] = []
    for quality in quality_candidates:
        outcome = await try_once(default_size, quality)
        if outcome["ok"]:
            detected_qualities.append(quality)
            continue
        allowed = _extract_allowed_values_from_error(outcome["error"], "quality")
        if allowed:
            detected_qualities = [q for q in quality_candidates if q in allowed] or allowed
            break

    if not detected_qualities:
        detected_qualities = [default_quality]
    detected_quality = detected_qualities[0]

    detected_sizes: List[str] = []
    for size in size_candidates:
        outcome = await try_once(size, detected_quality)
        if outcome["ok"]:
            detected_sizes.append(size)
            continue
        allowed = _extract_allowed_values_from_error(outcome["error"], "size")
        if allowed:
            detected_sizes = [s for s in size_candidates if s in allowed] or allowed
            break

    if not detected_sizes:
        detected_sizes = [default_size]

    payload = {
        "provider": provider_key,
        "model": model_name,
        "sizes": list(dict.fromkeys([s for s in detected_sizes if s])),
        "qualities": list(dict.fromkeys([q for q in detected_qualities if q])),
        "default_size": default_size,
        "default_quality": default_quality,
        "source": "probe+raw",
        "fetched_at": now_iso(),
    }

    async with ctx.image_options_lock:
        ctx.image_options_cache[cache_key] = {
            "fetched_monotonic": time.monotonic(),
            "payload": payload,
        }
    return payload


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    started = time.monotonic()
    request_id = request.headers.get("x-request-id") or uuid4().hex
    token = set_request_id(request_id)
    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001
        elapsed = round((time.monotonic() - started) * 1000.0, 3)
        ctx = _ctx(request)
        if ctx.settings.logging.enabled and ctx.settings.logging.log_http_requests:
            log_event(
                app_logger,
                event="http.request.failed",
                level=logging.ERROR,
                message="http request failed",
                method=request.method,
                path=request.url.path,
                query=str(request.url.query or ""),
                latency_ms=elapsed,
                error=str(exc),
                client=(request.client.host if request.client else ""),
            )
        await ctx.event_tracker.record(
            event_type="http.request.failed",
            level="ERROR",
            message="http request failed",
            request_id=request_id,
            data={
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query or ""),
                "latency_ms": elapsed,
                "error": str(exc),
                "client": (request.client.host if request.client else ""),
            },
        )
        status_code = 503
        payload = {
            "ok": False,
            "error": "request failed before response",
            "detail": str(exc),
            "request_id": request_id,
            "status_code": status_code,
            "error_type": "internal_error",
        }
        reset_request_id(token)
        return JSONResponse(status_code=status_code, content=payload)

    response.headers["x-request-id"] = request_id
    elapsed = round((time.monotonic() - started) * 1000.0, 3)
    ctx = _ctx(request)
    status_code = int(response.status_code)
    if ctx.settings.logging.enabled and ctx.settings.logging.log_http_requests:
        lvl = logging.INFO if status_code < 400 else logging.WARNING
        log_event(
            app_logger,
            event="http.request.completed",
            level=lvl,
            message="http request completed",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query or ""),
            status_code=status_code,
            latency_ms=elapsed,
            client=(request.client.host if request.client else ""),
            content_length=response.headers.get("content-length", ""),
        )
    await ctx.event_tracker.record(
        event_type="http.request.completed",
        level="INFO" if status_code < 400 else "WARNING",
        message="http request completed",
        request_id=request_id,
        data={
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query or ""),
            "status_code": status_code,
            "latency_ms": elapsed,
            "client": (request.client.host if request.client else ""),
            "content_length": response.headers.get("content-length", ""),
        },
    )
    reset_request_id(token)
    return response


def _check_token(
    token: Optional[str],
    expected: str,
    enabled: bool,
) -> None:
    if not enabled:
        return
    if not expected:
        raise HTTPException(status_code=503, detail="Auth enabled but token is empty")
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API token")


async def require_client_auth(
    request: Request,
    token_header: Optional[str] = Header(default=None),
) -> None:
    ctx = _ctx(request)
    cfg = ctx.settings.auth
    provided = request.headers.get(cfg.header_name) or token_header
    try:
        _check_token(provided, cfg.token, cfg.enabled)
    except HTTPException as exc:
        log_event(
            app_logger,
            event="auth.client.failed",
            level=logging.WARNING,
            message="client authentication failed",
            path=request.url.path,
            status_code=exc.status_code,
            reason=str(exc.detail),
        )
        await ctx.event_tracker.record(
            event_type="auth.client.failed",
            level="WARNING",
            message="client authentication failed",
            request_id=get_request_id(),
            data={
                "path": request.url.path,
                "status_code": int(exc.status_code),
                "reason": str(exc.detail),
            },
        )
        raise


async def require_admin_auth(
    request: Request,
    token_header: Optional[str] = Header(default=None),
) -> None:
    ctx = _ctx(request)
    cfg = ctx.settings.admin
    if not cfg.enabled:
        raise HTTPException(status_code=404, detail="Admin disabled")
    provided = request.headers.get(cfg.header_name) or token_header
    try:
        _check_token(provided, cfg.token, True)
    except HTTPException as exc:
        log_event(
            app_logger,
            event="auth.admin.failed",
            level=logging.WARNING,
            message="admin authentication failed",
            path=request.url.path,
            status_code=exc.status_code,
            reason=str(exc.detail),
        )
        await ctx.event_tracker.record(
            event_type="auth.admin.failed",
            level="WARNING",
            message="admin authentication failed",
            request_id=get_request_id(),
            data={
                "path": request.url.path,
                "status_code": int(exc.status_code),
                "reason": str(exc.detail),
            },
        )
        raise


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> Dict[str, Any]:
    ctx = _ctx(request)
    redis_ok = await ctx.guard.ping()
    providers = {name: True for name in ctx.registry.names()}
    return {"status": "ok", "providers": providers, "redis_ok": redis_ok}


@app.get("/v1/models", dependencies=[Depends(require_client_auth)])
async def list_models(request: Request) -> Dict[str, Any]:
    ctx = _ctx(request)
    out = []
    for name, adapter in ctx.registry.providers.items():
        if not hasattr(adapter, "list_models"):
            continue
        result = await adapter.list_models()
        out.append(
            {
                "provider": name,
                "ok": result.ok,
                "status_code": result.status_code,
                "latency_ms": round(result.latency_ms, 2),
                "payload": result.payload,
                "error": result.error,
            }
        )
    return {"object": "list", "data": out}


@app.get("/v1/models/catalog", dependencies=[Depends(require_client_auth)])
async def list_models_catalog(
    request: Request,
    refresh: bool = Query(default=False),
    providers: str | None = Query(default=None, description="Comma-separated providers: gemini,groq,pollinations"),
    capability: str | None = Query(default=None),
    model_type: str | None = Query(default=None, description="llm,embedding,image,audio_tts,audio_stt,multi,other"),
    modality: str | None = Query(default=None, description="text,image,audio,embedding,video"),
    include_preview: bool = Query(default=True),
    include_paid: bool = Query(default=True),
    search: str | None = Query(default=None),
    include_raw: bool = Query(default=False),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    catalog = await _refresh_models_catalog(ctx, refresh=refresh)
    provider_list = [x.strip().lower() for x in str(providers or "").split(",") if x.strip()]

    filtered_base = filter_catalog_models(
        list(catalog.get("items") or []),
        providers=provider_list,
        capability=capability,
        model_type=model_type,
        modality=modality,
        include_preview=include_preview,
        include_paid=include_paid,
        search=search,
    )
    filtered = [dict(row) for row in filtered_base]
    filtered.sort(key=lambda row: (str(row.get("provider") or ""), str(row.get("id") or "")))
    if not include_raw:
        for row in filtered:
            row.pop("raw", None)

    provider_status = [{k: v for k, v in row.items() if k != "items"} for row in list(catalog.get("provider_rows") or [])]

    return {
        "generated_at": now_iso(),
        "from_cache": bool(catalog.get("from_cache")),
        "fetched_at": catalog.get("fetched_at"),
        "fallback_applied": bool(catalog.get("fallback_applied")),
        "filters": {
            "providers": provider_list,
            "capability": capability or "",
            "model_type": model_type or "",
            "modality": modality or "",
            "include_preview": include_preview,
            "include_paid": include_paid,
            "search": search or "",
            "include_raw": include_raw,
        },
        "providers_status": provider_status,
        "summary": summarize_catalog_models(filtered),
        "count": len(filtered),
        "items": filtered,
    }


@app.get("/v1/models/catalog/summary", dependencies=[Depends(require_client_auth)])
async def list_models_catalog_summary(
    request: Request,
    refresh: bool = Query(default=False),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    catalog = await _refresh_models_catalog(ctx, refresh=refresh)
    items = list(catalog.get("items") or [])
    compact_items = [dict(item) for item in items]
    for row in compact_items:
        row.pop("raw", None)
    provider_status = [{k: v for k, v in row.items() if k != "items"} for row in list(catalog.get("provider_rows") or [])]

    return {
        "generated_at": now_iso(),
        "from_cache": bool(catalog.get("from_cache")),
        "fetched_at": catalog.get("fetched_at"),
        "fallback_applied": bool(catalog.get("fallback_applied")),
        "providers_status": provider_status,
        "summary": summarize_catalog_models(compact_items),
    }


@app.get("/v1/models/providers", dependencies=[Depends(require_client_auth)])
async def list_models_providers_status(
    request: Request,
    refresh: bool = Query(default=False),
    include_items: bool = Query(default=False),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    catalog = await _refresh_models_catalog(ctx, refresh=refresh)
    rows = list(catalog.get("provider_rows") or [])
    if not include_items:
        rows = [{k: v for k, v in row.items() if k != "items"} for row in rows]
    return {
        "generated_at": now_iso(),
        "from_cache": bool(catalog.get("from_cache")),
        "fetched_at": catalog.get("fetched_at"),
        "fallback_applied": bool(catalog.get("fallback_applied")),
        "count": len(rows),
        "items": rows,
    }


@app.get("/v1/images/options", dependencies=[Depends(require_client_auth)])
async def image_model_options(
    request: Request,
    provider: str = Query(..., description="Image provider, e.g. pollinations"),
    model: str = Query(..., description="Image model id"),
    refresh: bool = Query(default=False),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    payload = await _resolve_image_options(ctx, provider=provider, model=model, refresh=refresh)
    return payload


@app.post("/v1/chat/completions", dependencies=[Depends(require_client_auth)])
async def chat_completions(request: Request, body: Dict[str, Any]) -> Response:
    ctx = _ctx(request)
    stream_enabled = bool(body.get("stream"))
    options_raw = body.pop("x_router", None)
    options = RouterOptions.model_validate(options_raw) if options_raw else None

    if stream_enabled:
        stream_result = await ctx.router.dispatch_chat_stream(body, options)
        if not stream_result.get("ok"):
            return JSONResponse(status_code=_router_failure_status(ctx, stream_result, default=502), content=stream_result)
        stream_iter = stream_result.get("stream")
        if stream_iter is None:
            stream_result["ok"] = False
            stream_result["status_code"] = 502
            stream_result["error_type"] = "stream_unavailable"
            return JSONResponse(status_code=502, content=stream_result)

        winner = stream_result.get("winner") or {}
        headers = {
            "cache-control": "no-cache",
            "connection": "keep-alive",
            "x-uag-router-strategy": str(stream_result.get("strategy") or ""),
            "x-uag-router-mode": str(stream_result.get("mode") or ""),
            "x-uag-provider": str(winner.get("provider") or ""),
            "x-uag-model": str(winner.get("model") or ""),
        }
        return StreamingResponse(stream_iter, media_type="text/event-stream", headers=headers)

    result = await ctx.router.dispatch("chat.completions", body, options)
    if not result.get("ok"):
        return JSONResponse(status_code=_router_failure_status(ctx, result, default=502), content=result)

    winner = result.get("winner") or {}
    if winner.get("provider") == "gemini":
        normalized = ProviderResult(
            provider=winner.get("provider", "gemini"),
            capability=winner.get("capability", "chat.completions"),
            ok=bool(winner.get("ok", False)),
            status_code=int(winner.get("status_code", 200)),
            latency_ms=float(winner.get("latency_ms", 0.0)),
            payload=winner.get("payload"),
            error=str(winner.get("error", "")),
            model=str(winner.get("model", "")),
            headers=dict(winner.get("headers") or {}),
        )
        openai_payload = gemini_to_openai_chat(normalized)
        openai_payload["router"] = {
            "strategy": result.get("strategy"),
            "mode": result.get("mode"),
            "request_id": request.headers.get("x-request-id") or uuid4().hex,
        }
        return JSONResponse(status_code=200, content=openai_payload)

    payload = winner.get("payload")
    if isinstance(payload, (dict, list)):
        wrapped = dict(payload) if isinstance(payload, dict) else {"data": payload}
        wrapped["router"] = {
            "strategy": result.get("strategy"),
            "mode": result.get("mode"),
            "provider": winner.get("provider"),
            "latency_ms": winner.get("latency_ms"),
        }
        return JSONResponse(status_code=200, content=wrapped)

    return JSONResponse(status_code=200, content={"result": payload, "router": result})


@app.post("/v1/responses", dependencies=[Depends(require_client_auth)])
async def responses_api(request: Request, body: Dict[str, Any]) -> Response:
    ctx = _ctx(request)
    options_raw = body.pop("x_router", None)
    options = RouterOptions.model_validate(options_raw) if options_raw else None
    result = await ctx.router.dispatch("responses", body, options)
    return JSONResponse(status_code=200 if result.get("ok") else _router_failure_status(ctx, result, default=502), content=result)


@app.post("/v1/embeddings", dependencies=[Depends(require_client_auth)])
async def embeddings_api(request: Request, body: Dict[str, Any]) -> Response:
    ctx = _ctx(request)
    options_raw = body.pop("x_router", None)
    options = RouterOptions.model_validate(options_raw) if options_raw else None
    result = await ctx.router.dispatch("embeddings", body, options)
    return JSONResponse(status_code=200 if result.get("ok") else _router_failure_status(ctx, result, default=502), content=result)


@app.post("/v1/images/generations", dependencies=[Depends(require_client_auth)])
async def image_generations(request: Request, body: Dict[str, Any]) -> Response:
    ctx = _ctx(request)
    options_raw = body.pop("x_router", None)
    options = RouterOptions.model_validate(options_raw) if options_raw else None
    result = await ctx.router.dispatch("images.generations", body, options)
    return JSONResponse(status_code=200 if result.get("ok") else _router_failure_status(ctx, result, default=502), content=result)


@app.post("/v1/audio/transcriptions", dependencies=[Depends(require_client_auth)])
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    language: str | None = None,
) -> Response:
    ctx = _ctx(request)
    adapter = ctx.registry.providers.get("groq")
    if adapter is None or not hasattr(adapter, "stt"):
        raise HTTPException(status_code=503, detail="Groq STT provider is not enabled")

    data = await file.read()
    stt_timeout_sec = max(5.0, float(ctx.settings.router.parallel_timeout_sec))
    try:
        result = await asyncio.wait_for(
            adapter.stt(
                audio_bytes=data,
                filename=file.filename or "audio.ogg",
                content_type=file.content_type or "audio/ogg",
                language=language,
            ),
            timeout=stt_timeout_sec,
        )
    except asyncio.TimeoutError:
        result = ProviderResult(
            provider="groq",
            capability="audio.transcriptions",
            ok=False,
            status_code=504,
            latency_ms=0.0,
            payload=None,
            error=f"provider timeout after {stt_timeout_sec}s",
            model="",
        )
    await ctx.usage_tracker.record(
        capability="audio.transcriptions",
        priority=0,
        result=result,
    )
    await ctx.event_tracker.record(
        event_type="provider.direct.transcription",
        level="INFO" if result.ok else "WARNING",
        message="direct provider transcription completed",
        request_id=get_request_id(),
        data={
            "provider": result.provider,
            "model": result.model,
            "ok": bool(result.ok),
            "status_code": int(result.status_code),
            "latency_ms": round(float(result.latency_ms), 3),
            "filename": str(file.filename or ""),
        },
    )
    if result.ok:
        return JSONResponse(status_code=200, content=result.__dict__)
    status_code = max(400, min(599, int(result.status_code or 502)))
    payload = dict(result.__dict__)
    payload["status_code"] = status_code
    payload["error_type"] = (
        "all_rate_limited"
        if status_code == 429
        else ("upstream_timeout" if status_code in {408, 504} else "provider_error")
    )
    return JSONResponse(status_code=status_code, content=payload)


@app.post("/v1/audio/speech", dependencies=[Depends(require_client_auth)])
async def audio_speech(request: Request, body: Dict[str, Any]) -> Response:
    ctx = _ctx(request)
    options_raw = body.pop("x_router", None)
    options = RouterOptions.model_validate(options_raw) if options_raw else None
    result = await ctx.router.dispatch("audio.speech", body, options)
    if not result.get("ok"):
        return JSONResponse(status_code=_router_failure_status(ctx, result, default=502), content=result)

    winner = result.get("winner") or {}
    payload = winner.get("payload")
    if isinstance(payload, (bytes, bytearray)):
        headers = dict(winner.get("headers") or {})
        media_type = str(headers.get("content-type") or headers.get("Content-Type") or "audio/wav")
        return Response(
            content=bytes(payload),
            media_type=media_type,
            headers={
                "x-router-strategy": str(result.get("strategy") or ""),
                "x-router-mode": str(result.get("mode") or ""),
                "x-router-provider": str(winner.get("provider") or ""),
            },
        )

    return JSONResponse(status_code=200, content=result)


@app.post("/v1/orchestrate", dependencies=[Depends(require_client_auth)])
async def orchestrate(request: Request, body: OrchestrateRequest) -> Response:
    ctx = _ctx(request)
    result = await ctx.router.dispatch(body.capability, dict(body.payload), body.x_router)
    return JSONResponse(status_code=200 if result.get("ok") else _router_failure_status(ctx, result, default=502), content=result)


@app.get("/admin/router/stats", dependencies=[Depends(require_admin_auth)])
async def admin_router_stats(request: Request) -> Dict[str, Any]:
    ctx = _ctx(request)
    return ctx.router.stats()


@app.get("/admin/usage/overview", dependencies=[Depends(require_admin_auth)])
async def admin_usage_overview(
    request: Request,
    since_minutes: int = Query(default=60, ge=0, le=10080),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    return await ctx.usage_tracker.overview(since_minutes=since_minutes)


@app.get("/admin/usage/events", dependencies=[Depends(require_admin_auth)])
async def admin_usage_events(
    request: Request,
    since_minutes: int = Query(default=60, ge=0, le=10080),
    limit: int = Query(default=200, ge=1, le=5000),
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    capability: str | None = Query(default=None),
    key_id: str | None = Query(default=None),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    return await ctx.usage_tracker.events(
        provider=provider,
        model=model,
        key_id=key_id,
        capability=capability,
        since_minutes=since_minutes,
        limit=limit,
    )


@app.get("/admin/usage/aggregate", dependencies=[Depends(require_admin_auth)])
async def admin_usage_aggregate(
    request: Request,
    group_by: str = Query(default="provider"),
    since_minutes: int = Query(default=60, ge=0, le=10080),
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    capability: str | None = Query(default=None),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    try:
        return await ctx.usage_tracker.aggregate(
            group_by=group_by,
            since_minutes=since_minutes,
            provider=provider,
            model=model,
            capability=capability,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/admin/usage/providers", dependencies=[Depends(require_admin_auth)])
async def admin_usage_providers(
    request: Request,
    since_minutes: int = Query(default=60, ge=0, le=10080),
    capability: str | None = Query(default=None),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    return await ctx.usage_tracker.aggregate(
        group_by="provider",
        since_minutes=since_minutes,
        capability=capability,
    )


@app.get("/admin/usage/models", dependencies=[Depends(require_admin_auth)])
async def admin_usage_models(
    request: Request,
    since_minutes: int = Query(default=60, ge=0, le=10080),
    provider: str | None = Query(default=None),
    capability: str | None = Query(default=None),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    return await ctx.usage_tracker.aggregate(
        group_by="model",
        since_minutes=since_minutes,
        provider=provider,
        capability=capability,
    )


@app.get("/admin/usage/keys", dependencies=[Depends(require_admin_auth)])
async def admin_usage_keys(
    request: Request,
    since_minutes: int = Query(default=60, ge=0, le=10080),
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    capability: str | None = Query(default=None),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    return await ctx.usage_tracker.aggregate(
        group_by="key",
        since_minutes=since_minutes,
        provider=provider,
        model=model,
        capability=capability,
    )


@app.get("/admin/usage/key-limits/latest", dependencies=[Depends(require_admin_auth)])
async def admin_usage_key_limits_latest(
    request: Request,
    provider: str | None = Query(default=None),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    return await ctx.usage_tracker.key_limits_latest(provider=provider)


@app.get("/admin/logs/events", dependencies=[Depends(require_admin_auth)])
async def admin_logs_events(
    request: Request,
    since_minutes: int = Query(default=60, ge=0, le=10080),
    limit: int = Query(default=200, ge=1, le=5000),
    level: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    request_id: str | None = Query(default=None),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    return await ctx.event_tracker.events(
        since_minutes=since_minutes,
        limit=limit,
        level=level,
        event_type=event_type,
        request_id=request_id,
    )


@app.get("/admin/logs/http-stats", dependencies=[Depends(require_admin_auth)])
async def admin_logs_http_stats(
    request: Request,
    group_by: str = Query(default="path"),
    since_minutes: int = Query(default=60, ge=0, le=10080),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    try:
        return await ctx.event_tracker.http_request_stats(group_by=group_by, since_minutes=since_minutes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/admin/logs/summary", dependencies=[Depends(require_admin_auth)])
async def admin_logs_summary(
    request: Request,
    since_minutes: int = Query(default=60, ge=0, le=10080),
) -> Dict[str, Any]:
    ctx = _ctx(request)
    return {
        "generated_at": now_iso(),
        "since_minutes": since_minutes,
        "events": await ctx.event_tracker.summary(since_minutes=since_minutes),
        "usage_overview": await ctx.usage_tracker.overview(since_minutes=since_minutes),
        "usage_by_provider": await ctx.usage_tracker.aggregate(group_by="provider", since_minutes=since_minutes),
        "usage_by_model": await ctx.usage_tracker.aggregate(group_by="model", since_minutes=since_minutes),
        "usage_by_key": await ctx.usage_tracker.aggregate(group_by="key", since_minutes=since_minutes),
        "router_scores": ctx.router.stats(),
    }


@app.get("/admin", include_in_schema=False)
@app.get("/admin/", include_in_schema=False)
async def admin_panel_index() -> Response:
    index_path = ADMIN_UI_ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Admin panel UI not found")
    return FileResponse(index_path)


@app.get("/admin/{asset_path:path}", include_in_schema=False)
async def admin_panel_assets(asset_path: str) -> Response:
    cleaned = (asset_path or "").strip().lstrip("/")
    # Keep API/Admin functional routes untouched; this route is for static assets and SPA fallback only.
    blocked_prefixes = (
        "router/",
        "usage/",
        "logs/",
    )
    if any(cleaned.startswith(prefix) for prefix in blocked_prefixes):
        raise HTTPException(status_code=404, detail="Not Found")

    if not cleaned:
        target = ADMIN_UI_ROOT / "index.html"
    else:
        target = (ADMIN_UI_ROOT / cleaned).resolve()
        if not str(target).startswith(str(ADMIN_UI_ROOT.resolve())):
            raise HTTPException(status_code=403, detail="Forbidden")
        if not target.exists() or not target.is_file():
            target = ADMIN_UI_ROOT / "index.html"

    if not target.exists():
        raise HTTPException(status_code=404, detail="Admin panel UI not found")
    return FileResponse(target)


@app.get("/studio", include_in_schema=False)
@app.get("/studio/", include_in_schema=False)
async def studio_index() -> Response:
    index_path = STUDIO_UI_ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Studio UI not found")
    return FileResponse(index_path)


@app.get("/studio/{asset_path:path}", include_in_schema=False)
async def studio_assets(asset_path: str) -> Response:
    cleaned = (asset_path or "").strip().lstrip("/")
    if not cleaned:
        target = STUDIO_UI_ROOT / "index.html"
    else:
        target = (STUDIO_UI_ROOT / cleaned).resolve()
        if not str(target).startswith(str(STUDIO_UI_ROOT.resolve())):
            raise HTTPException(status_code=403, detail="Forbidden")
        if not target.exists() or not target.is_file():
            target = STUDIO_UI_ROOT / "index.html"

    if not target.exists():
        raise HTTPException(status_code=404, detail="Studio UI not found")
    return FileResponse(target)


@app.websocket("/ws/admin")
async def ws_admin(websocket: WebSocket):
    await websocket.accept()
    ctx: AppState = websocket.app.state.ctx
    ws_logger = logging.getLogger("uag.ws.admin")
    request_id = websocket.query_params.get("request_id") or uuid4().hex
    token = set_request_id(request_id)

    cfg = ctx.settings.admin
    if not cfg.enabled:
        await websocket.send_json({"ok": False, "error": "Admin disabled"})
        await websocket.close(code=4404)
        reset_request_id(token)
        return

    auth_token = websocket.query_params.get("token") or websocket.headers.get(cfg.header_name)
    try:
        _check_token(auth_token, cfg.token, True)
    except HTTPException:
        await websocket.send_json({"ok": False, "error": "Unauthorized"})
        await websocket.close(code=4401)
        reset_request_id(token)
        return

    if ctx.settings.logging.enabled and ctx.settings.logging.log_websocket:
        log_event(
            ws_logger,
            event="ws.admin.connected",
            level=logging.INFO,
            message="admin websocket connected",
            path="/ws/admin",
        )
    await ctx.event_tracker.record(
        event_type="ws.admin.connected",
        level="INFO",
        message="admin websocket connected",
        request_id=request_id,
        data={"path": "/ws/admin"},
    )

    scopes: set[str] = {"summary", "events", "usage", "http", "router"}
    since_minutes = _to_int(websocket.query_params.get("since_minutes"), 60, 0, 10080)
    events_limit = _to_int(websocket.query_params.get("limit"), 200, 1, 1000)
    events_level: str | None = None
    events_type: str | None = None
    events_request_id: str | None = None
    http_group_by = "path"
    after_seq = _to_int(websocket.query_params.get("after_seq"), 0, 0, 10_000_000)

    async def build_payload(*, incremental: bool) -> Dict[str, Any]:
        nonlocal after_seq
        payload: Dict[str, Any] = {
            "type": "tick",
            "request_id": request_id,
            "server_time": now_iso(),
            "scopes": sorted(scopes),
        }
        if "events" in scopes:
            events = await ctx.event_tracker.events(
                since_minutes=since_minutes,
                level=events_level,
                event_type=events_type,
                request_id=events_request_id,
                after_seq=after_seq if incremental else 0,
                limit=events_limit,
            )
            payload["events"] = events
            after_seq = max(after_seq, int(events.get("latest_seq") or 0))
        if "summary" in scopes:
            payload["summary"] = await ctx.event_tracker.summary(since_minutes=since_minutes)
        if "http" in scopes:
            payload["http"] = await ctx.event_tracker.http_request_stats(group_by=http_group_by, since_minutes=since_minutes)
        if "usage" in scopes:
            payload["usage_overview"] = await ctx.usage_tracker.overview(since_minutes=since_minutes)
            payload["usage_by_provider"] = await ctx.usage_tracker.aggregate(group_by="provider", since_minutes=since_minutes)
            payload["usage_by_model"] = await ctx.usage_tracker.aggregate(group_by="model", since_minutes=since_minutes)
            payload["usage_by_key"] = await ctx.usage_tracker.aggregate(group_by="key", since_minutes=since_minutes)
        if "router" in scopes:
            payload["router_scores"] = ctx.router.stats()
        return payload

    hello_payload = await build_payload(incremental=False)
    hello_payload["type"] = "hello"
    await websocket.send_json(hello_payload)

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                try:
                    incoming = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "error": "Invalid JSON payload"})
                    continue
                if not isinstance(incoming, dict):
                    await websocket.send_json({"type": "error", "error": "Payload must be object"})
                    continue

                msg_type = str(incoming.get("type") or "").strip().lower()
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong", "server_time": now_iso()})
                    continue
                if msg_type == "subscribe":
                    raw_scopes = incoming.get("scopes")
                    if isinstance(raw_scopes, list):
                        normalized = {str(s).strip().lower() for s in raw_scopes if str(s).strip()}
                        valid = {"summary", "events", "usage", "http", "router"}
                        selected = normalized.intersection(valid)
                        if selected:
                            scopes = selected
                    await websocket.send_json({"type": "subscribed", "scopes": sorted(scopes)})
                    continue
                if msg_type == "filters":
                    since_minutes = _to_int(incoming.get("since_minutes"), since_minutes, 0, 10080)
                    events_limit = _to_int(incoming.get("limit"), events_limit, 1, 1000)
                    level_raw = str(incoming.get("level") or "").strip().upper()
                    events_level = level_raw or None
                    type_raw = str(incoming.get("event_type") or "").strip()
                    events_type = type_raw or None
                    req_raw = str(incoming.get("request_id_filter") or "").strip()
                    events_request_id = req_raw or None
                    group_raw = str(incoming.get("http_group_by") or http_group_by).strip()
                    if group_raw in {"path", "method", "status_code", "status_class", "path_method"}:
                        http_group_by = group_raw
                    after_seq = _to_int(incoming.get("after_seq"), after_seq, 0, 10_000_000)
                    await websocket.send_json(
                        {
                            "type": "filters.applied",
                            "since_minutes": since_minutes,
                            "limit": events_limit,
                            "level": events_level,
                            "event_type": events_type,
                            "request_id_filter": events_request_id,
                            "http_group_by": http_group_by,
                            "after_seq": after_seq,
                        }
                    )
                    continue
            except TimeoutError:
                pass

            tick_payload = await build_payload(incremental=True)
            await websocket.send_json(tick_payload)
    except WebSocketDisconnect:
        if ctx.settings.logging.enabled and ctx.settings.logging.log_websocket:
            log_event(
                ws_logger,
                event="ws.admin.disconnected",
                level=logging.INFO,
                message="admin websocket disconnected",
                path="/ws/admin",
            )
        await ctx.event_tracker.record(
            event_type="ws.admin.disconnected",
            level="INFO",
            message="admin websocket disconnected",
            request_id=request_id,
            data={"path": "/ws/admin"},
        )
        reset_request_id(token)
        return
    except Exception as exc:  # noqa: BLE001
        if ctx.settings.logging.enabled and ctx.settings.logging.log_websocket:
            log_event(
                ws_logger,
                event="ws.admin.error",
                level=logging.ERROR,
                message="admin websocket error",
                error=str(exc),
            )
        await ctx.event_tracker.record(
            event_type="ws.admin.error",
            level="ERROR",
            message="admin websocket error",
            request_id=request_id,
            data={"error": str(exc)},
        )
        reset_request_id(token)
        raise


@app.websocket("/ws/llm")
async def ws_llm(websocket: WebSocket):
    await websocket.accept()
    ctx: AppState = websocket.app.state.ctx
    ws_logger = logging.getLogger("uag.ws")
    request_id = websocket.query_params.get("request_id") or uuid4().hex
    token = set_request_id(request_id)

    # token can be sent as query param or header
    auth_token = websocket.query_params.get("token") or websocket.headers.get(ctx.settings.auth.header_name)
    try:
        _check_token(auth_token, ctx.settings.auth.token, ctx.settings.auth.enabled)
    except HTTPException:
        if ctx.settings.logging.enabled and ctx.settings.logging.log_websocket:
            log_event(
                ws_logger,
                event="ws.auth.failed",
                level=logging.WARNING,
                message="websocket authentication failed",
                path="/ws/llm",
            )
        await ctx.event_tracker.record(
            event_type="ws.auth.failed",
            level="WARNING",
            message="websocket authentication failed",
            request_id=request_id,
            data={"path": "/ws/llm"},
        )
        await websocket.send_json({"ok": False, "error": "Unauthorized"})
        await websocket.close(code=4401)
        reset_request_id(token)
        return

    if ctx.settings.logging.enabled and ctx.settings.logging.log_websocket:
        log_event(
            ws_logger,
            event="ws.connected",
            level=logging.INFO,
            message="websocket connected",
            path="/ws/llm",
        )
    await ctx.event_tracker.record(
        event_type="ws.connected",
        level="INFO",
        message="websocket connected",
        request_id=request_id,
        data={"path": "/ws/llm"},
    )

    try:
        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json({"ok": False, "error": "Invalid JSON"})
                continue

            payload = dict(data.get("payload") or data)
            options_raw = data.get("x_router")
            options = RouterOptions.model_validate(options_raw) if options_raw else None
            if ctx.settings.logging.enabled and ctx.settings.logging.log_websocket:
                log_event(
                    ws_logger,
                    event="ws.message.received",
                    level=logging.INFO,
                    message="websocket message received",
                    payload_keys=sorted(list(payload.keys())),
                )
            result = await ctx.router.dispatch("chat.completions", payload, options)
            await websocket.send_json(result)
            await ctx.event_tracker.record(
                event_type="ws.message.handled",
                level="INFO" if result.get("ok") else "WARNING",
                message="websocket message handled",
                request_id=request_id,
                data={
                    "ok": bool(result.get("ok")),
                    "strategy": str(result.get("strategy") or ""),
                    "mode": str(result.get("mode") or ""),
                    "latency_ms": float(result.get("latency_ms") or 0.0),
                },
            )
    except WebSocketDisconnect:
        if ctx.settings.logging.enabled and ctx.settings.logging.log_websocket:
            log_event(
                ws_logger,
                event="ws.disconnected",
                level=logging.INFO,
                message="websocket disconnected",
                path="/ws/llm",
            )
        await ctx.event_tracker.record(
            event_type="ws.disconnected",
            level="INFO",
            message="websocket disconnected",
            request_id=request_id,
            data={"path": "/ws/llm"},
        )
        reset_request_id(token)
        return
    except Exception as exc:  # noqa: BLE001
        if ctx.settings.logging.enabled and ctx.settings.logging.log_websocket:
            log_event(
                ws_logger,
                event="ws.error",
                level=logging.ERROR,
                message="websocket error",
                error=str(exc),
            )
        await ctx.event_tracker.record(
            event_type="ws.error",
            level="ERROR",
            message="websocket error",
            request_id=request_id,
            data={"error": str(exc)},
        )
        reset_request_id(token)
        raise
