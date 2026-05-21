# Ajil Unified AI Gateway (UAG)

Production-oriented FastAPI gateway that exposes a single AI API surface over multiple providers:
- Gemini
- Groq
- Pollinations

It gives you one unified service for chat, vision, embeddings, image generation, STT, TTS, routing, failover, load balancing, observability, and interactive UIs.

## Table of Contents
1. [What This Project Solves](#what-this-project-solves)
2. [Feature Map](#feature-map)
3. [Architecture](#architecture)
4. [Repository Layout](#repository-layout)
5. [Quick Start](#quick-start)
6. [Configuration (`.env`)](#configuration-env)
7. [Authentication](#authentication)
8. [Routing and Model Selection](#routing-and-model-selection)
9. [Streaming](#streaming)
10. [HTTP API Endpoints](#http-api-endpoints)
11. [WebSocket APIs](#websocket-apis)
12. [Studio UI](#studio-ui)
13. [Admin Panel and Observability](#admin-panel-and-observability)
14. [Load Testing and Artifacts](#load-testing-and-artifacts)
15. [Provider-Specific Notes](#provider-specific-notes)
16. [Troubleshooting](#troubleshooting)
17. [Production Checklist](#production-checklist)
18. [FAQ](#faq)

## What This Project Solves

Instead of integrating with each provider separately, you integrate once with UAG and get:
- Unified OpenAI-like API shape for most common AI tasks
- Multi-provider fallback and request routing
- Multi-key usage with retries and rotation in provider modules
- Endpoint-level and provider-attempt telemetry
- Real-time admin dashboard and usage analytics
- End-user Studio for manual testing across capabilities

## Feature Map

### Core capabilities
- `chat.completions` (text + vision-style image input)
- `responses`
- `embeddings`
- `images.generations`
- `audio.transcriptions` (STT)
- `audio.speech` (TTS)
- generic `/v1/orchestrate`

### Routing capabilities
- Routing strategies:
  - `fallback_chain`
  - `parallel_race`
  - `aggregate`
- Routing modes:
  - `latency_first`
  - `limit_safe`
  - `quality_first`
- Candidate shaping from both:
  - payload `model`
  - `x_router` options
- Priority-based model list support (`priority: 0` = highest)
- Candidate deduplication and capability filtering
- Local fallback response for chat/responses during upstream instability (non-429 scenarios)

### Observability capabilities
- In-memory usage tracker with filters and aggregates
- In-memory structured event tracker with HTTP stats
- Per-provider/per-model/per-key aggregates
- Latest seen upstream rate-limit headers by key
- Router scorecard (`latency`, `failures`, `rate_limited`, `total_calls`)
- Realtime admin websocket feed

### UX capabilities
- `/studio`: interactive multi-tab test workbench
- `/admin`: live admin/monitoring dashboard with websocket updates

## Architecture

`Client -> UAG (FastAPI) -> Router Engine -> Provider Adapter -> Provider Module -> Upstream API`

- `unified_gateway` is the API runtime and routing brain.
- `modules/*` are imported provider proxy modules with key rotation/retry logic.
- Redis is used for shared gateway-side rate-limit guard.

## Repository Layout

```text
.
├── unified_gateway/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── router/engine.py
│   │   ├── providers/
│   │   ├── admin_ui/
│   │   └── studio_ui/
│   └── tests/
├── modules/
│   ├── gemini_proxy/        # provider module (submodule target)
│   ├── groq_proxy/          # provider module (submodule target)
│   └── pollinations_proxy/  # provider module (submodule target)
├── scripts/
│   ├── compose_up.sh
│   └── load_test_gateway.py
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── README.md
```

## Quick Start

### A) Docker Compose (recommended)

```bash
./scripts/compose_up.sh
```

What it does:
1. Creates `.env` from `.env.example` if missing
2. Builds gateway image
3. Starts `gateway` + `redis`
4. Optionally starts `proxy2080` profile if enabled in `.env`

Default URLs:
- API: `http://127.0.0.1:8080`
- Swagger: `http://127.0.0.1:8080/docs`
- Studio: `http://127.0.0.1:8080/studio`
- Admin: `http://127.0.0.1:8080/admin`

Useful commands:
```bash
docker compose ps
docker compose logs -f gateway
docker compose down
```

### B) Local run (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r unified_gateway/requirements.txt
cp .env.example .env
uvicorn unified_gateway.app.main:app --host 0.0.0.0 --port 8080 --reload
```

### C) Initialize submodules (if needed)

```bash
git submodule update --init --recursive
```

## Configuration (`.env`)

All runtime config is centralized in root `.env`.

Use `.env.example` as the authoritative template.

### 1) App and docs
- `UAG_APP_HOST`
- `UAG_APP_PORT`
- `UAG_APP_LOG_LEVEL`
- `UAG_APP_DOCS_ENABLED`

### 2) Client auth
- `UAG_AUTH_ENABLED`
- `UAG_AUTH_TOKEN`
- `UAG_AUTH_HEADER_NAME`

### 3) Admin auth
- `UAG_ADMIN_ENABLED`
- `UAG_ADMIN_TOKEN`
- `UAG_ADMIN_HEADER_NAME`

### 4) Router defaults
- `UAG_ROUTER_DEFAULT_STRATEGY`
- `UAG_ROUTER_DEFAULT_MODE`
- `UAG_ROUTER_PARALLEL_TIMEOUT_SEC`
- `UAG_ROUTER_MAX_CANDIDATES`
- `UAG_ROUTER_FALLBACK_MAX_ATTEMPTS`
- `UAG_ROUTER_FALLBACK_ATTEMPT_TIMEOUT_SEC`
- `UAG_ROUTER_FALLBACK_ROUND_BACKOFF_SEC`
- `UAG_ROUTER_RETRYABLE_STATUS_CODES`
- `UAG_ROUTER_STRICT_RATE_LIMIT_ERRORS_ONLY`

### 5) Redis and gateway guard
- `UAG_REDIS_REQUIRED`
- `UAG_REDIS_URL`
- `UAG_REDIS_KEY_PREFIX`
- `UAG_REDIS_DEFAULT_LIMIT_PER_MINUTE`
- `UAG_REDIS_URL_DOCKER`
- `UAG_REDIS_HOST_PORT`

### 6) Logging and telemetry buffers
- `UAG_LOG_ENABLED`
- `UAG_LOG_JSON`
- `UAG_LOG_HTTP_REQUESTS`
- `UAG_LOG_ROUTER_DISPATCH`
- `UAG_LOG_PROVIDER_ATTEMPTS`
- `UAG_LOG_WEBSOCKET`
- `UAG_LOG_MAX_EVENTS`
- `UAG_USAGE_MAX_EVENTS`

### 7) Global outbound proxy
- `UAG_PROXY_ENABLED`
- `UAG_PROXY_URL`
- `UAG_PROXY_URL_DOCKER`
- `UAG_PROXY_USE_COMPOSE_SERVICE`

### 8) Docker build network/proxy
- `UAG_DOCKER_BUILD_NETWORK`
- `UAG_BUILD_HTTP_PROXY`
- `UAG_BUILD_HTTPS_PROXY`
- `UAG_BUILD_ALL_PROXY`
- `UAG_BUILD_NO_PROXY`
- `UAG_PIP_INDEX_URL`
- `UAG_PIP_EXTRA_INDEX_URL`

### 9) Gemini provider
- `UAG_GEMINI_ENABLED`
- `UAG_GEMINI_MODE` (`gemini_direct | cloudflare_worker`)
- `UAG_GEMINI_BASE_URL`
- `UAG_GEMINI_API_VERSION`
- `UAG_GEMINI_DEFAULT_MODEL`
- `UAG_GEMINI_API_KEYS`
- `UAG_GEMINI_RETRY_ON_429`
- `UAG_GEMINI_RETRY_ON_5XX`
- `UAG_GEMINI_MAX_RETRIES_PER_KEY`
- `UAG_GEMINI_MAX_RETRIES_ON_5XX`
- `UAG_GEMINI_RETRY_BACKOFF_SEC`
- `UAG_GEMINI_COOLOFF_SEC`
- `UAG_GEMINI_MIN_INTERVAL_SEC`
- Cloudflare worker mode:
  - `UAG_GEMINI_WORKER_BASE_URLS`
  - `UAG_GEMINI_WORKER_ROUTE_PREFIX`
  - `UAG_GEMINI_WORKER_AUTH_TOKEN`
  - `UAG_GEMINI_WORKER_AUTH_HEADER_NAME`

### 10) Groq provider
- `UAG_GROQ_ENABLED`
- `UAG_GROQ_BASE_URL`
- `UAG_GROQ_API_KEYS`
- `UAG_GROQ_RETRY_ON_429`
- `UAG_GROQ_RETRY_ON_5XX`
- `UAG_GROQ_MAX_RETRIES_PER_KEY`
- `UAG_GROQ_MAX_RETRIES_ON_5XX`
- `UAG_GROQ_RETRY_BACKOFF_SEC`
- `UAG_GROQ_COOLOFF_SEC`
- `UAG_GROQ_MIN_INTERVAL_SEC`
- STT defaults:
  - `UAG_GROQ_STT_PRIMARY_MODEL`
  - `UAG_GROQ_STT_FALLBACK_MODEL`
  - `UAG_GROQ_STT_LANGUAGE`
  - `UAG_GROQ_STT_TEMPERATURE`
  - `UAG_GROQ_STT_RESPONSE_FORMAT`
  - `UAG_GROQ_STT_PROMPT`
- TTS defaults:
  - `UAG_GROQ_TTS_DEFAULT_MODEL`
  - `UAG_GROQ_TTS_DEFAULT_VOICE`
  - `UAG_GROQ_TTS_DEFAULT_RESPONSE_FORMAT`

### 11) Pollinations provider
- `UAG_POLLINATIONS_ENABLED`
- `UAG_POLLINATIONS_BASE_URL`
- `UAG_POLLINATIONS_API_KEYS`
- `UAG_POLLINATIONS_DEFAULT_IMAGE_MODEL`
- `UAG_POLLINATIONS_MAX_ATTEMPTS_PER_REQUEST`
- `UAG_POLLINATIONS_RETRY_STATUS_CODES`
- `UAG_POLLINATIONS_RETRY_BACKOFF_SEC`
- `UAG_POLLINATIONS_KEY_COOLDOWN_SEC`
- `UAG_POLLINATIONS_LOCAL_PLACEHOLDER_ON_FAILURE`
- image defaults:
  - `UAG_IMAGE_DEFAULT_N`
  - `UAG_IMAGE_DEFAULT_SIZE`
  - `UAG_IMAGE_DEFAULT_QUALITY`
  - `UAG_IMAGE_DEFAULT_RESPONSE_FORMAT`

### Backward-compatible env aliases
Supported fallback keys also include:
- `GROQ_API_KEYS`, `GROQ_API_KEY`
- `GEMINI_API_KEYS`
- `POLLINATIONS_API_KEYS`, `POLLINATIONS_API_KEY`

## Authentication

### Client endpoints
If `UAG_AUTH_ENABLED=true`, send:
```http
x-api-token: <UAG_AUTH_TOKEN>
```
(or your custom `UAG_AUTH_HEADER_NAME`)

### Admin endpoints
If `UAG_ADMIN_ENABLED=true`, send:
```http
x-admin-token: <UAG_ADMIN_TOKEN>
```
(or your custom `UAG_ADMIN_HEADER_NAME`)

## Routing and Model Selection

### `model` field
`model` supports all forms below:
- string:
  - `"gemini/gemini-2.5-flash"`
- object:
  - `{"provider":"gemini","model":"gemini-2.5-flash","priority":0}`
- list of objects (recommended):
  - explicit ordered candidates with priority

Example:
```json
"model": [
  {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0},
  {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1}
]
```

Priority rule:
- lower number = higher priority (`0` is highest)

### `x_router` field
Per-request routing policy:
```json
"x_router": {
  "providers": ["groq", "gemini"],
  "models": ["groq/llama-3.3-70b-versatile", "gemini/gemini-2.5-flash"],
  "model_preferences": [
    {"provider":"groq","model":"llama-3.3-70b-versatile","priority":0}
  ],
  "strategy": "fallback_chain",
  "mode": "limit_safe",
  "max_attempts": 6,
  "timeout_sec": 25,
  "result_policy": "envelope_with_candidates"
}
```

### Strategies
- `fallback_chain`: sequential attempts by sorted candidate order
- `parallel_race`: fire candidates concurrently, return first success
- `aggregate`: execute all candidates and return envelope with all results

### Modes
- `latency_first`: prioritize lower observed latency
- `limit_safe`: prioritize candidates with fewer rate-limit/failure signals
- `quality_first`: provider preference ordering optimized for quality

## Streaming

### Chat streaming over HTTP SSE
`POST /v1/chat/completions` supports `"stream": true`.

- Response content type: `text/event-stream`
- Returns `data: ...` chunks and final `data: [DONE]`
- Winner metadata is set in headers:
  - `x-uag-provider`
  - `x-uag-model`
  - `x-uag-router-strategy`
  - `x-uag-router-mode`

Example:
```bash
curl -N -X POST "http://127.0.0.1:8080/v1/chat/completions" \
  -H "x-api-token: replace_with_client_token" \
  -H "content-type: application/json" \
  -d '{
    "model": [
      {"provider":"groq","model":"llama-3.3-70b-versatile","priority":0},
      {"provider":"gemini","model":"gemini-2.5-flash","priority":1}
    ],
    "messages": [{"role":"user","content":"سلام"}],
    "stream": true,
    "x_router": {"strategy":"fallback_chain","mode":"limit_safe"}
  }'
```

Notes:
- Streaming path is chat-focused.
- If all upstreams fail with non-rate-limit instability, local fallback stream may be emitted.

## HTTP API Endpoints

### Health
- `GET /health`

### Models and catalog
- `GET /v1/models`
  - raw per-provider model payload
- `GET /v1/models/catalog`
  - normalized, filterable model catalog
  - filters: `providers`, `capability`, `model_type`, `modality`, `include_preview`, `include_paid`, `search`, `include_raw`, `refresh`
- `GET /v1/models/catalog/summary`
- `GET /v1/models/providers`
- `GET /v1/images/options?provider=...&model=...`
  - returns allowed image `size` and `quality` choices for selected model/provider

### Inference
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/audio/transcriptions` (multipart form)
- `POST /v1/audio/speech`
- `POST /v1/orchestrate`

### Admin data APIs
- `GET /admin/router/stats`
- `GET /admin/usage/overview`
- `GET /admin/usage/events`
- `GET /admin/usage/aggregate`
- `GET /admin/usage/providers`
- `GET /admin/usage/models`
- `GET /admin/usage/keys`
- `GET /admin/usage/key-limits/latest`
- `GET /admin/logs/events`
- `GET /admin/logs/http-stats`
- `GET /admin/logs/summary`

### UI routes
- `GET /studio` (and static assets)
- `GET /admin` (and static assets)

## WebSocket APIs

### `WS /ws/llm`
Chat-completion websocket endpoint.

Auth:
- query `token`, or header with `UAG_AUTH_HEADER_NAME`

Message format:
```json
{
  "payload": {
    "model": [{"provider":"gemini","model":"gemini-2.5-flash","priority":0}],
    "messages": [{"role":"user","content":"hello"}]
  },
  "x_router": {
    "strategy": "fallback_chain",
    "mode": "latency_first"
  }
}
```

Response:
- router result envelope JSON

### `WS /ws/admin`
Realtime monitoring stream for admin panel.

Capabilities:
- initial `hello` payload with summary/events/usage/http/router
- periodic `tick` updates
- command messages:
  - `ping`
  - `subscribe` (scopes)
  - `filters` (since, level, event type, request-id filter, http group)

## Studio UI

Open: `http://127.0.0.1:8080/studio`

Tabs:
- Chat
- Vision (image understanding)
- Responses
- Image generation
- Text-to-Speech
- Speech-to-Text
- Embeddings
- Orchestrate
- Model Explorer

Key behaviors:
- model list pulled from live catalog
- vision tab shows only vision-capable models
- image tab fetches real `size/quality` options via `/v1/images/options`
- chat and vision support streaming toggle (SSE)
- markdown rendering in assistant output with RTL detection for Persian content

## Admin Panel and Observability

Open: `http://127.0.0.1:8080/admin`

Capabilities:
- websocket live feed from `/ws/admin`
- event stream filtering by level/type/request-id
- HTTP stats by group (`path`, `method`, `status_code`, `status_class`, `path_method`)
- usage aggregation by:
  - `provider`
  - `model`
  - `key`
  - `provider_model`
  - `provider_model_key`
  - `capability`
- key rate-limit header snapshots
- router score table

Data tracked per provider attempt:
- success/failure
- status code
- latency
- provider/model
- key identity (masked/slot when available)
- usage token info when upstream exposes it

## Load Testing and Artifacts

Script: `scripts/load_test_gateway.py`

What it can test:
- chat, responses, embeddings, orchestrate
- STT (`/v1/audio/transcriptions`)
- TTS (`/v1/audio/speech`)
- image generation (`/v1/images/generations`) with capped request count

Features:
- randomized scenarios with configurable weights
- configurable concurrency and total requests
- provider/model discovery from `/v1/models`
- Persian prompts by default
- stores request/response artifacts in results directory

Example:
```bash
python3 scripts/load_test_gateway.py \
  --base-url http://127.0.0.1:8080 \
  --token "$UAG_AUTH_TOKEN" \
  --token-header "${UAG_AUTH_HEADER_NAME:-x-api-token}" \
  --admin-token "$UAG_ADMIN_TOKEN" \
  --admin-token-header "${UAG_ADMIN_HEADER_NAME:-x-admin-token}" \
  --total 100 \
  --concurrency 12
```

Quick test:
```bash
python3 scripts/load_test_gateway.py --total 20 --concurrency 5 --seed 7 --no-trust-env
```

## Provider-Specific Notes

### Groq
- STT uses Whisper models through `/v1/audio/transcriptions`
- TTS uses Groq-supported speech models via `/v1/audio/speech`
- Some TTS models may require terms acceptance in Groq console (`model_terms_required`)

### Gemini
- Supported in direct mode and Cloudflare worker mode
- Vision input is mapped to Gemini content parts
- Streaming chat is normalized to OpenAI-style chunk events

### Pollinations
- Used for image generation capability
- can run with retries/cooldowns
- optional local placeholder behavior on failure can be enabled

## Troubleshooting

### Docker build fails with DNS error
Symptom: `Temporary failure in name resolution` during `pip install`

Fix:
1. Set `UAG_DOCKER_BUILD_NETWORK=host` in `.env`
2. Re-run `./scripts/compose_up.sh`

If your network needs proxy during build:
- set `UAG_BUILD_HTTP_PROXY`, `UAG_BUILD_HTTPS_PROXY`, `UAG_BUILD_ALL_PROXY`

### Port conflict on Redis (`6379 already in use`)
- change `UAG_REDIS_HOST_PORT` in `.env`
- or stop local Redis occupying 6379

### Local websocket issues when proxy env is set
- configure `NO_PROXY=127.0.0.1,localhost`

### Repeated upstream timeout errors
- increase timeout budgets:
  - `UAG_ROUTER_PARALLEL_TIMEOUT_SEC`
  - `UAG_ROUTER_FALLBACK_ATTEMPT_TIMEOUT_SEC`
- ensure outbound connectivity/proxy is healthy
- verify provider keys are valid and not restricted

## Production Checklist

1. Set strong auth secrets:
- `UAG_AUTH_TOKEN`
- `UAG_ADMIN_TOKEN`

2. Keep `.env` out of git.

3. Restrict admin UI/API exposure by network policy.

4. Monitor:
- `/admin/logs/summary`
- `/admin/usage/overview`
- `/admin/router/stats`

5. Configure proxy and timeout settings to match your network.

6. Size telemetry buffers based on traffic:
- `UAG_USAGE_MAX_EVENTS`
- `UAG_LOG_MAX_EVENTS`

## FAQ

### What is the difference between `GROQ_API_KEYS` and `GROQ_API_KEY`?
- `GROQ_API_KEYS`: comma-separated list of keys (preferred)
- `GROQ_API_KEY`: single key fallback (legacy compatibility)

Gateway-native equivalent is `UAG_GROQ_API_KEYS`.

### Why do I see both `model` and `x_router.models`?
- `model` is the request’s primary model declaration (supports priority objects)
- `x_router.models` is an explicit router-level candidate list
- UAG merges candidates from both and deduplicates

### Can providers stream?
- Chat streaming is available through `POST /v1/chat/completions` with `stream=true`
- Studio chat/vision tabs can display stream output live

### Is this project a single service from outside?
Yes. Clients call one gateway API. Internally it orchestrates providers and modules.

---

If you need a client SDK-style guide (Python/Node snippets for all endpoints), add it as a next step and keep this README as the operational reference.
