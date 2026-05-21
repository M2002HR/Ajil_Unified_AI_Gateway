from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from .config import Settings, get_settings
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


def _ctx(request: Request) -> AppState:
    return request.app.state.ctx


def _to_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        out = int(str(value))
    except (TypeError, ValueError):
        out = default
    return max(min_value, min(max_value, out))


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
        reset_request_id(token)
        raise

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


@app.post("/v1/chat/completions", dependencies=[Depends(require_client_auth)])
async def chat_completions(request: Request, body: Dict[str, Any]) -> Response:
    ctx = _ctx(request)
    options_raw = body.pop("x_router", None)
    options = RouterOptions.model_validate(options_raw) if options_raw else None

    result = await ctx.router.dispatch("chat.completions", body, options)
    if not result.get("ok"):
        return JSONResponse(status_code=502, content=result)

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
    return JSONResponse(status_code=200 if result.get("ok") else 502, content=result)


@app.post("/v1/embeddings", dependencies=[Depends(require_client_auth)])
async def embeddings_api(request: Request, body: Dict[str, Any]) -> Response:
    ctx = _ctx(request)
    options_raw = body.pop("x_router", None)
    options = RouterOptions.model_validate(options_raw) if options_raw else None
    result = await ctx.router.dispatch("embeddings", body, options)
    return JSONResponse(status_code=200 if result.get("ok") else 502, content=result)


@app.post("/v1/images/generations", dependencies=[Depends(require_client_auth)])
async def image_generations(request: Request, body: Dict[str, Any]) -> Response:
    ctx = _ctx(request)
    options_raw = body.pop("x_router", None)
    options = RouterOptions.model_validate(options_raw) if options_raw else None
    result = await ctx.router.dispatch("images.generations", body, options)
    return JSONResponse(status_code=200 if result.get("ok") else 502, content=result)


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
    result = await adapter.stt(
        audio_bytes=data,
        filename=file.filename or "audio.ogg",
        content_type=file.content_type or "audio/ogg",
        language=language,
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
    return JSONResponse(status_code=200 if result.ok else 502, content=result.__dict__)


@app.post("/v1/audio/speech", dependencies=[Depends(require_client_auth)])
async def audio_speech(request: Request, body: Dict[str, Any]) -> Response:
    ctx = _ctx(request)
    options_raw = body.pop("x_router", None)
    options = RouterOptions.model_validate(options_raw) if options_raw else None
    result = await ctx.router.dispatch("audio.speech", body, options)
    if not result.get("ok"):
        return JSONResponse(status_code=502, content=result)

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
    return JSONResponse(status_code=200 if result.get("ok") else 502, content=result)


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
