from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


def _to_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _to_csv(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    docs_enabled: bool = True


class AuthConfig(BaseModel):
    enabled: bool = True
    token: str = ""
    header_name: str = "x-api-token"


class AdminConfig(BaseModel):
    enabled: bool = True
    token: str = ""
    header_name: str = "x-admin-token"


class RouterConfig(BaseModel):
    default_strategy: Literal["fallback_chain", "parallel_race", "aggregate"] = "fallback_chain"
    default_mode: Literal["latency_first", "limit_safe", "quality_first"] = "latency_first"
    parallel_timeout_sec: float = 25.0
    max_candidates: int = 6
    fallback_max_attempts: int = 18
    fallback_attempt_timeout_sec: float = 12.0
    fallback_round_backoff_sec: float = 0.2
    retryable_status_codes: List[int] = Field(default_factory=lambda: [408, 409, 425, 429, 500, 502, 503, 504])
    strict_rate_limit_errors_only: bool = True


class ProxyConfig(BaseModel):
    enabled: bool = False
    url: str = "socks5://127.0.0.1:2080"

    @staticmethod
    def normalize(url: str) -> str:
        out = (url or "").strip()
        if out.startswith("socks://"):
            out = "socks5://" + out[len("socks://") :]
        return out


class RedisConfig(BaseModel):
    required: bool = False
    url: str = "redis://redis:6379/0"
    key_prefix: str = "uag"
    default_limit_per_minute: int = 120


class UsageConfig(BaseModel):
    max_events: int = 20000


class LoggingConfig(BaseModel):
    enabled: bool = True
    json_logs: bool = True
    log_http_requests: bool = True
    log_router_dispatch: bool = True
    log_provider_attempts: bool = True
    log_websocket: bool = True
    max_events: int = 50000


class GeminiProviderConfig(BaseModel):
    enabled: bool = True
    mode: Literal["gemini_direct", "cloudflare_worker"] = "gemini_direct"
    base_url: str = "https://generativelanguage.googleapis.com"
    api_version: str = "v1beta"
    default_model: str = "gemini-2.5-flash"
    api_keys: List[str] = Field(default_factory=list)
    worker_base_urls: List[str] = Field(default_factory=list)
    worker_route_prefix: str = "/gemini"
    worker_auth_token: str = ""
    worker_auth_header_name: str = "x-worker-auth"
    retry_on_429: bool = True
    retry_on_5xx: bool = True
    max_retries_per_key: int = 2
    max_retries_on_5xx: int = 4
    retry_backoff_sec: float = 0.35
    cooloff_sec: float = 20.0
    min_interval_sec: float = 0.0


class GroqProviderConfig(BaseModel):
    enabled: bool = True
    base_url: str = "https://api.groq.com/openai/v1"
    api_keys: List[str] = Field(default_factory=list)
    stt_primary_model: str = "whisper-large-v3-turbo"
    stt_fallback_model: str = "whisper-large-v3"
    stt_language: str = "fa"
    stt_temperature: float = 0.0
    stt_response_format: str = "verbose_json"
    stt_prompt: str = ""
    tts_default_model: str = "canopylabs/orpheus-v1-english"
    tts_default_voice: str = "diana"
    tts_default_response_format: str = "wav"
    retry_on_429: bool = True
    retry_on_5xx: bool = True
    max_retries_per_key: int = 2
    max_retries_on_5xx: int = 4
    retry_backoff_sec: float = 0.35
    cooloff_sec: float = 20.0
    min_interval_sec: float = 0.0


class PollinationsProviderConfig(BaseModel):
    enabled: bool = True
    base_url: str = "https://gen.pollinations.ai"
    api_keys: List[str] = Field(default_factory=list)
    default_image_model: str = "flux"
    use_proxy_2080: bool = False
    proxy_2080_url: str = "socks5://127.0.0.1:2080"
    trust_env_proxy: bool = False
    max_attempts_per_request: int = 6
    retry_status_codes: List[int] = Field(default_factory=lambda: [402, 429, 500, 502, 503, 504])
    retry_backoff_sec: float = 0.35
    cooldown_sec: float = 20.0
    local_placeholder_on_failure: bool = True
    image_default_n: int = 1
    image_default_size: str = "1024x1024"
    image_default_quality: str = "medium"
    image_default_response_format: str = "b64_json"


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    usage: UsageConfig = Field(default_factory=UsageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    gemini: GeminiProviderConfig = Field(default_factory=GeminiProviderConfig)
    groq: GroqProviderConfig = Field(default_factory=GroqProviderConfig)
    pollinations: PollinationsProviderConfig = Field(default_factory=PollinationsProviderConfig)


def _load_env() -> None:
    env_file = os.getenv("UAG_ENV_FILE", ".env")
    load_dotenv(dotenv_path=Path(env_file), override=False)


def load_settings() -> Settings:
    _load_env()

    data: Dict[str, Any] = {
        "app": {
            "host": os.getenv("UAG_APP_HOST", "0.0.0.0"),
            "port": int(os.getenv("UAG_APP_PORT", "8080")),
            "log_level": os.getenv("UAG_APP_LOG_LEVEL", "INFO"),
            "docs_enabled": _to_bool(os.getenv("UAG_APP_DOCS_ENABLED"), True),
        },
        "auth": {
            "enabled": _to_bool(os.getenv("UAG_AUTH_ENABLED"), True),
            "token": os.getenv("UAG_AUTH_TOKEN", ""),
            "header_name": os.getenv("UAG_AUTH_HEADER_NAME", "x-api-token"),
        },
        "admin": {
            "enabled": _to_bool(os.getenv("UAG_ADMIN_ENABLED"), True),
            "token": os.getenv("UAG_ADMIN_TOKEN", ""),
            "header_name": os.getenv("UAG_ADMIN_HEADER_NAME", "x-admin-token"),
        },
        "router": {
            "default_strategy": os.getenv("UAG_ROUTER_DEFAULT_STRATEGY", "fallback_chain"),
            "default_mode": os.getenv("UAG_ROUTER_DEFAULT_MODE", "latency_first"),
            "parallel_timeout_sec": float(os.getenv("UAG_ROUTER_PARALLEL_TIMEOUT_SEC", "25")),
            "max_candidates": int(os.getenv("UAG_ROUTER_MAX_CANDIDATES", "6")),
            "fallback_max_attempts": int(os.getenv("UAG_ROUTER_FALLBACK_MAX_ATTEMPTS", "18")),
            "fallback_attempt_timeout_sec": float(os.getenv("UAG_ROUTER_FALLBACK_ATTEMPT_TIMEOUT_SEC", "12")),
            "fallback_round_backoff_sec": float(os.getenv("UAG_ROUTER_FALLBACK_ROUND_BACKOFF_SEC", "0.2")),
            "retryable_status_codes": [
                int(v) for v in _to_csv(os.getenv("UAG_ROUTER_RETRYABLE_STATUS_CODES", "408,409,425,429,500,502,503,504"))
            ],
            "strict_rate_limit_errors_only": _to_bool(os.getenv("UAG_ROUTER_STRICT_RATE_LIMIT_ERRORS_ONLY"), True),
        },
        "proxy": {
            "enabled": _to_bool(os.getenv("UAG_PROXY_ENABLED"), False),
            "url": os.getenv("UAG_PROXY_URL", "socks5://127.0.0.1:2080"),
        },
        "redis": {
            "required": _to_bool(os.getenv("UAG_REDIS_REQUIRED"), False),
            "url": os.getenv("UAG_REDIS_URL", "redis://redis:6379/0"),
            "key_prefix": os.getenv("UAG_REDIS_KEY_PREFIX", "uag"),
            "default_limit_per_minute": int(os.getenv("UAG_REDIS_DEFAULT_LIMIT_PER_MINUTE", "120")),
        },
        "usage": {
            "max_events": int(os.getenv("UAG_USAGE_MAX_EVENTS", "20000")),
        },
        "logging": {
            "enabled": _to_bool(os.getenv("UAG_LOG_ENABLED"), True),
            "json_logs": _to_bool(os.getenv("UAG_LOG_JSON"), True),
            "log_http_requests": _to_bool(os.getenv("UAG_LOG_HTTP_REQUESTS"), True),
            "log_router_dispatch": _to_bool(os.getenv("UAG_LOG_ROUTER_DISPATCH"), True),
            "log_provider_attempts": _to_bool(os.getenv("UAG_LOG_PROVIDER_ATTEMPTS"), True),
            "log_websocket": _to_bool(os.getenv("UAG_LOG_WEBSOCKET"), True),
            "max_events": int(os.getenv("UAG_LOG_MAX_EVENTS", "50000")),
        },
        "gemini": {
            "enabled": _to_bool(os.getenv("UAG_GEMINI_ENABLED"), True),
            "mode": os.getenv("UAG_GEMINI_MODE", "gemini_direct"),
            "base_url": os.getenv("UAG_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
            "api_version": os.getenv("UAG_GEMINI_API_VERSION", "v1beta"),
            "default_model": os.getenv("UAG_GEMINI_DEFAULT_MODEL", "gemini-2.5-flash"),
            "api_keys": _to_csv(os.getenv("UAG_GEMINI_API_KEYS")),
            "worker_base_urls": _to_csv(os.getenv("UAG_GEMINI_WORKER_BASE_URLS")),
            "worker_route_prefix": os.getenv("UAG_GEMINI_WORKER_ROUTE_PREFIX", "/gemini"),
            "worker_auth_token": os.getenv("UAG_GEMINI_WORKER_AUTH_TOKEN", ""),
            "worker_auth_header_name": os.getenv("UAG_GEMINI_WORKER_AUTH_HEADER_NAME", "x-worker-auth"),
            "retry_on_429": _to_bool(os.getenv("UAG_GEMINI_RETRY_ON_429"), True),
            "retry_on_5xx": _to_bool(os.getenv("UAG_GEMINI_RETRY_ON_5XX"), True),
            "max_retries_per_key": int(os.getenv("UAG_GEMINI_MAX_RETRIES_PER_KEY", "2")),
            "max_retries_on_5xx": int(os.getenv("UAG_GEMINI_MAX_RETRIES_ON_5XX", "4")),
            "retry_backoff_sec": float(os.getenv("UAG_GEMINI_RETRY_BACKOFF_SEC", "0.35")),
            "cooloff_sec": float(os.getenv("UAG_GEMINI_COOLOFF_SEC", "20")),
            "min_interval_sec": float(os.getenv("UAG_GEMINI_MIN_INTERVAL_SEC", "0")),
        },
        "groq": {
            "enabled": _to_bool(os.getenv("UAG_GROQ_ENABLED"), True),
            "base_url": os.getenv("UAG_GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            "api_keys": _to_csv(os.getenv("UAG_GROQ_API_KEYS") or os.getenv("GROQ_API_KEYS")),
            "stt_primary_model": os.getenv("UAG_GROQ_STT_PRIMARY_MODEL", "whisper-large-v3-turbo"),
            "stt_fallback_model": os.getenv("UAG_GROQ_STT_FALLBACK_MODEL", "whisper-large-v3"),
            "stt_language": os.getenv("UAG_GROQ_STT_LANGUAGE", "fa"),
            "stt_temperature": float(os.getenv("UAG_GROQ_STT_TEMPERATURE", "0")),
            "stt_response_format": os.getenv("UAG_GROQ_STT_RESPONSE_FORMAT", "verbose_json"),
            "stt_prompt": os.getenv("UAG_GROQ_STT_PROMPT", ""),
            "tts_default_model": os.getenv("UAG_GROQ_TTS_DEFAULT_MODEL", "canopylabs/orpheus-v1-english"),
            "tts_default_voice": os.getenv("UAG_GROQ_TTS_DEFAULT_VOICE", "diana"),
            "tts_default_response_format": os.getenv("UAG_GROQ_TTS_DEFAULT_RESPONSE_FORMAT", "wav"),
            "retry_on_429": _to_bool(os.getenv("UAG_GROQ_RETRY_ON_429"), True),
            "retry_on_5xx": _to_bool(os.getenv("UAG_GROQ_RETRY_ON_5XX"), True),
            "max_retries_per_key": int(os.getenv("UAG_GROQ_MAX_RETRIES_PER_KEY", "2")),
            "max_retries_on_5xx": int(os.getenv("UAG_GROQ_MAX_RETRIES_ON_5XX", "4")),
            "retry_backoff_sec": float(os.getenv("UAG_GROQ_RETRY_BACKOFF_SEC", "0.35")),
            "cooloff_sec": float(os.getenv("UAG_GROQ_COOLOFF_SEC", "20")),
            "min_interval_sec": float(os.getenv("UAG_GROQ_MIN_INTERVAL_SEC", "0")),
        },
        "pollinations": {
            "enabled": _to_bool(os.getenv("UAG_POLLINATIONS_ENABLED"), True),
            "base_url": os.getenv("UAG_POLLINATIONS_BASE_URL", "https://gen.pollinations.ai"),
            "api_keys": _to_csv(os.getenv("UAG_POLLINATIONS_API_KEYS")),
            "default_image_model": os.getenv("UAG_POLLINATIONS_DEFAULT_IMAGE_MODEL", "flux"),
            "use_proxy_2080": _to_bool(os.getenv("UAG_POLLINATIONS_USE_PROXY_2080"), False),
            "proxy_2080_url": os.getenv("UAG_POLLINATIONS_PROXY_2080_URL", "socks5://127.0.0.1:2080"),
            "trust_env_proxy": _to_bool(os.getenv("UAG_POLLINATIONS_TRUST_ENV_PROXY"), False),
            "max_attempts_per_request": int(os.getenv("UAG_POLLINATIONS_MAX_ATTEMPTS_PER_REQUEST", "6")),
            "retry_status_codes": [int(v) for v in _to_csv(os.getenv("UAG_POLLINATIONS_RETRY_STATUS_CODES", "402,429,500,502,503,504"))],
            "retry_backoff_sec": float(os.getenv("UAG_POLLINATIONS_RETRY_BACKOFF_SEC", "0.35")),
            "cooldown_sec": float(os.getenv("UAG_POLLINATIONS_KEY_COOLDOWN_SEC", "20")),
            "local_placeholder_on_failure": _to_bool(os.getenv("UAG_POLLINATIONS_LOCAL_PLACEHOLDER_ON_FAILURE"), True),
            "image_default_n": int(os.getenv("UAG_IMAGE_DEFAULT_N", "1")),
            "image_default_size": os.getenv("UAG_IMAGE_DEFAULT_SIZE", "1024x1024"),
            "image_default_quality": os.getenv("UAG_IMAGE_DEFAULT_QUALITY", "medium"),
            "image_default_response_format": os.getenv("UAG_IMAGE_DEFAULT_RESPONSE_FORMAT", "b64_json"),
        },
    }

    if not data["gemini"]["api_keys"]:
        data["gemini"]["api_keys"] = _to_csv(os.getenv("GEMINI_API_KEYS"))
    if not data["groq"]["api_keys"]:
        data["groq"]["api_keys"] = _to_csv(os.getenv("GROQ_API_KEY"))
    if not data["pollinations"]["api_keys"]:
        data["pollinations"]["api_keys"] = _to_csv(os.getenv("POLLINATIONS_API_KEYS") or os.getenv("POLLINATIONS_API_KEY"))

    settings = Settings.model_validate(data)
    settings.proxy.url = ProxyConfig.normalize(settings.proxy.url)
    if settings.proxy.enabled:
        settings.pollinations.use_proxy_2080 = True
        settings.pollinations.proxy_2080_url = settings.proxy.url
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
