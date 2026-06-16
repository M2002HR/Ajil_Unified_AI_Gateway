# Unified AI Gateway (Reference)

This file is a compact reference for the unified gateway.

For full documentation, use:
- `README.md` at the repository root

## Quick Start

```bash
./scripts/compose_up.sh
```

## Core Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`
- `POST /v1/orchestrate`
- `GET /admin`
- `GET /admin/router/stats`
- `GET /admin/usage/overview`
- `GET /admin/usage/events`
- `GET /admin/usage/aggregate`
- `GET /admin/usage/providers`
- `GET /admin/usage/models`
- `GET /admin/usage/keys`
- `GET /admin/usage/key-limits/latest`
- `GET /admin/logs/summary`
- `GET /admin/logs/events`
- `GET /admin/logs/http-stats`
- `WS /ws/llm`
- `WS /ws/admin`

## Priority-based `model` Format

```json
{
  "model": [
    {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0},
    {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1}
  ],
  "messages": [{"role": "user", "content": "hello"}],
  "x_router": {
    "strategy": "fallback_chain",
    "mode": "latency_first"
  }
}
```

Rules:
- Lower `priority` means higher priority
- On failure, the next priority candidate is attempted
- Equal priorities can start in any order

## Optional Outbound Proxy

```env
UAG_PROXY_ENABLED=true
UAG_PROXY_URL=socks5://127.0.0.1:2080
```

In Docker Compose, gateway uses:
- `UAG_PROXY_URL_DOCKER` (default `http://proxy2080:2080`)

## Usage Telemetry Buffer

```env
UAG_USAGE_MAX_EVENTS=20000
```

## Logging Config

```env
UAG_LOG_ENABLED=true
UAG_LOG_JSON=true
UAG_LOG_HTTP_REQUESTS=true
UAG_LOG_ROUTER_DISPATCH=true
UAG_LOG_PROVIDER_ATTEMPTS=true
UAG_LOG_WEBSOCKET=true
UAG_LOG_MAX_EVENTS=50000
```

## Build Network/Proxy (Docker Image Build Stage)

```env
UAG_DOCKER_BUILD_NETWORK=default
UAG_BUILD_HTTP_PROXY=
UAG_BUILD_HTTPS_PROXY=
UAG_BUILD_ALL_PROXY=
UAG_BUILD_NO_PROXY=127.0.0.1,localhost
UAG_PIP_INDEX_URL=
UAG_PIP_EXTRA_INDEX_URL=
```

If you get `Temporary failure in name resolution` while building:
- set `UAG_DOCKER_BUILD_NETWORK=host` (Linux)
- or provide build-time proxy values above

`/admin/usage/aggregate` supports:
- `provider`
- `model`
- `key`
- `provider_model`
- `provider_model_key`
- `capability`
