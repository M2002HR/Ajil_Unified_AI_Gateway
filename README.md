# Ajil Unified AI Gateway

A production-oriented **Unified AI Gateway** built with FastAPI.

It exposes a single API surface while routing requests across multiple providers:
- Gemini
- Groq
- Pollinations

The gateway handles provider selection, failover, and load-balancing policies internally so clients can integrate once.

---

## Key Features

- Single unified API for chat, responses, embeddings, image generation, STT, TTS, and orchestration
- Multi-provider routing with 3 strategies:
  - `fallback_chain`
  - `parallel_race`
  - `aggregate`
- **Priority-based model list** in payload (`model` can be a list of objects with `priority`)
- Per-request provider/model override through `x_router`
- Centralized `.env` configuration
- Optional outbound proxy support
- One-command Docker Compose startup
- Admin usage observability APIs (per-key, per-provider, per-model, capability, and combined groupings)
- Built-in realtime Admin Control Panel (`/admin`) with WebSocket streaming, filters, and interactive charts

---

## Architecture

`Client -> Unified Gateway (FastAPI) -> Provider Adapters -> Upstream APIs`

Main components:
- `unified_gateway/`: gateway core
- `modules/`: provider modules (Gemini/Groq/Pollinations)
- `redis`: shared rate-limit state
- `proxy2080` (optional): outbound proxy service in Compose

---

## Quick Start (One Command)

```bash
./scripts/compose_up.sh
```

What this does:
1. Creates `.env` from `.env.example` if missing
2. Builds the gateway image
3. Starts `gateway + redis`
4. Starts `proxy2080` too if `UAG_PROXY_ENABLED=true`

After startup:
- API: `http://127.0.0.1:8080`
- Swagger: `http://127.0.0.1:8080/docs`
- Admin Panel: `http://127.0.0.1:8080/admin`

Useful commands:
```bash
docker compose ps
docker compose logs -f gateway
docker compose down
```

---

## Environment Configuration

Primary settings in root `.env`:

### App/Auth/Admin
- `UAG_APP_HOST`, `UAG_APP_PORT`, `UAG_APP_LOG_LEVEL`
- `UAG_AUTH_ENABLED`, `UAG_AUTH_TOKEN`, `UAG_AUTH_HEADER_NAME`
- `UAG_ADMIN_ENABLED`, `UAG_ADMIN_TOKEN`, `UAG_ADMIN_HEADER_NAME`

### Router
- `UAG_ROUTER_DEFAULT_STRATEGY`: `fallback_chain | parallel_race | aggregate`
- `UAG_ROUTER_DEFAULT_MODE`: `latency_first | limit_safe | quality_first`
- `UAG_ROUTER_PARALLEL_TIMEOUT_SEC`
- `UAG_ROUTER_MAX_CANDIDATES`

### Redis
- `UAG_REDIS_REQUIRED`
- `UAG_REDIS_URL`
- `UAG_REDIS_KEY_PREFIX`
- `UAG_REDIS_DEFAULT_LIMIT_PER_MINUTE`

### Usage Telemetry
- `UAG_USAGE_MAX_EVENTS`: in-memory rolling buffer size for request telemetry

### Logging / Observability
- `UAG_LOG_ENABLED`
- `UAG_LOG_JSON`
- `UAG_LOG_HTTP_REQUESTS`
- `UAG_LOG_ROUTER_DISPATCH`
- `UAG_LOG_PROVIDER_ATTEMPTS`
- `UAG_LOG_WEBSOCKET`
- `UAG_LOG_MAX_EVENTS`: in-memory rolling buffer size for structured event logs

### Global Outbound Proxy (optional)
- `UAG_PROXY_ENABLED`
- `UAG_PROXY_URL` (supports `socks5://...` and `http://...`)
- `UAG_PROXY_USE_COMPOSE_SERVICE` (`true` to auto-start bundled `proxy2080` profile)

### Provider Keys
- Gemini:
  - `UAG_GEMINI_MODE`: `gemini_direct | cloudflare_worker`
  - `UAG_GEMINI_API_KEYS`
  - `UAG_GEMINI_WORKER_BASE_URLS` (worker mode)
- Groq:
  - `UAG_GROQ_API_KEYS`
- Pollinations:
  - `UAG_POLLINATIONS_API_KEYS`

### Compose-specific overrides
- `UAG_REDIS_URL_DOCKER` (default `redis://redis:6379/0`)
- `UAG_PROXY_URL_DOCKER` (default `http://proxy2080:3128`)

---

## API Endpoints

### Public
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`
- `POST /v1/orchestrate`
- `WS /ws/llm`

### Admin
- `GET /admin` (web UI entrypoint)
- `GET /admin/{asset_path}` (panel assets + SPA fallback)
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
- `WS /ws/admin`

Admin endpoints require admin token when enabled.

---

## Admin Control Panel

Open:
- `http://127.0.0.1:8080/admin`

Login inputs:
- Admin token: `UAG_ADMIN_TOKEN`
- Header name (for REST calls): `UAG_ADMIN_HEADER_NAME` (default `x-admin-token`)

Panel capabilities:
- Realtime websocket feed from `WS /ws/admin`
- Live event stream with level/type/request-id/search filters
- HTTP analytics (grouped by path/method/status class/code/path+method)
- Usage analytics (provider/model/key/capability and combined groupings)
- Router quality/latency/failure insights
- Latest key rate-limit header visibility

Charting:
- Uses Apache ECharts for interactive drill-down, zoom, and live updates

---

## Authentication

When `UAG_AUTH_ENABLED=true`, send:

```http
x-api-token: <UAG_AUTH_TOKEN>
```

---

## `model` vs `x_router`

- `model`: primary model input (OpenAI-style field)
- `x_router`: gateway routing policy for providers/models/strategy/mode

### Recommended advanced format for `model`

`model` can be a list of objects with explicit priority:

```json
"model": [
  {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0},
  {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1}
]
```

Priority rules:
- Lower `priority` means higher precedence
- `0` is higher priority than `1`
- In `fallback_chain`, higher-priority candidates are tried first
- On failure, gateway tries next priority candidate
- If priorities are equal, either candidate may be tried first

If both `model` and `x_router.models` are provided, both are used as candidates (with deduplication).

---

## Request Examples

### 1) Chat with priority-based fallback

```bash
curl -sS -X POST "http://127.0.0.1:8080/v1/chat/completions" \
  -H "x-api-token: replace_with_client_token" \
  -H "content-type: application/json" \
  -d '{
    "model": [
      {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0},
      {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1}
    ],
    "messages": [{"role": "user", "content": "hello"}],
    "x_router": {
      "strategy": "fallback_chain",
      "mode": "latency_first"
    }
  }'
```

### 2) Embeddings

```bash
curl -sS -X POST "http://127.0.0.1:8080/v1/embeddings" \
  -H "x-api-token: replace_with_client_token" \
  -H "content-type: application/json" \
  -d '{
    "model": "gemini/gemini-embedding-001",
    "input": "hello world",
    "x_router": {"providers": ["gemini"], "strategy": "fallback_chain"}
  }'
```

### 3) STT (Whisper)

```bash
curl -sS -X POST "http://127.0.0.1:8080/v1/audio/transcriptions" \
  -H "x-api-token: replace_with_client_token" \
  -F "file=@/path/to/audio.wav" \
  -F "language=fa"
```

### 4) Image generation

```bash
curl -sS -X POST "http://127.0.0.1:8080/v1/images/generations" \
  -H "x-api-token: replace_with_client_token" \
  -H "content-type: application/json" \
  -d '{
    "model": "pollinations/flux",
    "prompt": "minimal black icon of a cat",
    "size": "512x512",
    "quality": "low",
    "response_format": "url",
    "x_router": {"providers": ["pollinations"], "strategy": "fallback_chain"}
  }'
```

### 5) WebSocket

```text
ws://127.0.0.1:8080/ws/llm?token=<UAG_AUTH_TOKEN>
```

### 6) Admin Usage Overview

```bash
curl -sS "http://127.0.0.1:8080/admin/usage/overview?since_minutes=60" \
  -H "x-admin-token: replace_with_admin_token"
```

---

## Randomized Load Test

A randomized multi-endpoint load test script is available at:
- `scripts/load_test_gateway.py`

It tests:
- `/v1/chat/completions`
- `/v1/responses`
- `/v1/embeddings`
- `/v1/orchestrate`
- `/v1/audio/transcriptions`
- `/v1/audio/speech`
- `/v1/images/generations` (capped; default low count)

Default behavior:
- Total requests: `100`
- Random scenario mix
- Image requests capped to `4` by default
- Model discovery via `/v1/models`
- Gemini model selection prefers `gemma-4*`/`gemma*` when available
- Live terminal progress + final summary + admin snapshot

Run:

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

Useful options:
- `--max-image-requests 4` (default)
- `--max-speech-requests 12`
- `--max-stt-requests 10`
- `--seed 42` for reproducible randomization
- `--verbose` for per-request logs
- `--no-trust-env` to ignore proxy env vars during local tests

Example quick check:

```bash
python3 scripts/load_test_gateway.py --total 20 --concurrency 5 --seed 7 --no-trust-env
```

### 7) Admin Aggregate by Key

```bash
curl -sS "http://127.0.0.1:8080/admin/usage/aggregate?group_by=key&since_minutes=120" \
  -H "x-admin-token: replace_with_admin_token"
```

Supported `group_by` values:
- `provider`
- `model`
- `key`
- `provider_model`
- `provider_model_key`
- `capability`

### 8) Admin Raw Events Filtered by Provider + Model

```bash
curl -sS "http://127.0.0.1:8080/admin/usage/events?provider=groq&model=llama-3.3-70b-versatile&limit=100" \
  -H "x-admin-token: replace_with_admin_token"
```

### 9) Latest Captured Rate-Limit Headers per Key

```bash
curl -sS "http://127.0.0.1:8080/admin/usage/key-limits/latest?provider=groq" \
  -H "x-admin-token: replace_with_admin_token"
```

### 10) Log Summary (errors, success-rate, request latency)

```bash
curl -sS "http://127.0.0.1:8080/admin/logs/summary?since_minutes=60" \
  -H "x-admin-token: replace_with_admin_token"
```

### 11) Recent Structured Events

```bash
curl -sS "http://127.0.0.1:8080/admin/logs/events?since_minutes=60&limit=100" \
  -H "x-admin-token: replace_with_admin_token"
```

### 12) HTTP Stats by Group

```bash
curl -sS "http://127.0.0.1:8080/admin/logs/http-stats?group_by=path&since_minutes=60" \
  -H "x-admin-token: replace_with_admin_token"
```

---

## Operational Notes

1. **Whisper is STT, not TTS**
- Use Whisper with `/v1/audio/transcriptions`
- Use TTS models with `/v1/audio/speech`

2. **Groq TTS model terms**
- `model_terms_required` means model terms must be accepted in Groq console first.

3. **Pollinations timeouts / DNS issues**
- Enable outbound proxy:
  - `UAG_PROXY_ENABLED=true`
  - In Compose: `UAG_PROXY_URL_DOCKER=http://proxy2080:3128`

4. **WebSocket + local proxy env**
- If `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` are set globally, local WS handshakes can fail.
- Set `NO_PROXY=127.0.0.1,localhost` for local testing.

5. **How key/provider/model usage is measured**
- The gateway records telemetry for each provider attempt (including fallback/parallel attempts), not just the final winner.
- Per-key identity is derived from upstream proxy headers (`x-proxy-key-mask` / `x-proxy-key-slot`) when available.
- Latest rate-limit info comes from captured response headers (`x-ratelimit-*`, `ratelimit-*`, `retry-after`).
- Token usage is extracted from provider payloads (`usage` and Gemini `usageMetadata`) when present.

6. **Docker build DNS/proxy failures (pip install)**
- Symptom: `Temporary failure in name resolution` during image build.
- On Linux, set `UAG_DOCKER_BUILD_NETWORK=host` in `.env` and run `./scripts/compose_up.sh` again.
- If your environment requires outbound proxy during build, set:
  - `UAG_BUILD_HTTP_PROXY`
  - `UAG_BUILD_HTTPS_PROXY`
  - `UAG_BUILD_ALL_PROXY`
  - optionally `UAG_BUILD_NO_PROXY`
- Optional custom Python index mirrors:
  - `UAG_PIP_INDEX_URL`
  - `UAG_PIP_EXTRA_INDEX_URL`

---

## Local Development (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r unified_gateway/requirements.txt
cp .env.example .env
uvicorn unified_gateway.app.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## Project Structure

```text
unified_gateway/
  app/
    main.py
    config.py
    router/engine.py
    providers/
    state/rate_limit.py
  tests/
modules/
  gemini_proxy/
  groq_proxy/
  pollinations_proxy/
scripts/
  compose_up.sh
Dockerfile
docker-compose.yml
```

---

## Security

- Do not commit `.env`
- Use strong values for `UAG_AUTH_TOKEN` and `UAG_ADMIN_TOKEN`
- Restrict access to admin endpoints in production

---

## Current Status

- Unified gateway is implemented and operational
- One-command Docker Compose flow is available
- Priority-based model fallback is implemented and tested
- Optional global outbound proxy is supported
