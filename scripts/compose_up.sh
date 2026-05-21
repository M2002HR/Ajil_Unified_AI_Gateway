#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "[compose_up] .env created from .env.example"
fi

# Read proxy flag from .env (default false)
PROXY_ENABLED="$(awk -F= '/^UAG_PROXY_ENABLED=/{print tolower($2)}' .env | tail -n1)"
PROXY_ENABLED="${PROXY_ENABLED:-false}"
PROXY_USE_COMPOSE_SERVICE="$(awk -F= '/^UAG_PROXY_USE_COMPOSE_SERVICE=/{print tolower($2)}' .env | tail -n1)"
PROXY_USE_COMPOSE_SERVICE="${PROXY_USE_COMPOSE_SERVICE:-false}"
BUILD_NETWORK="$(awk -F= '/^UAG_DOCKER_BUILD_NETWORK=/{print $2}' .env | tail -n1)"
BUILD_NETWORK="${BUILD_NETWORK:-default}"

echo "[compose_up] Build network: ${BUILD_NETWORK}"

if [[ ( "$PROXY_ENABLED" == "true" || "$PROXY_ENABLED" == "1" || "$PROXY_ENABLED" == "yes" || "$PROXY_ENABLED" == "on" ) && ( "$PROXY_USE_COMPOSE_SERVICE" == "true" || "$PROXY_USE_COMPOSE_SERVICE" == "1" || "$PROXY_USE_COMPOSE_SERVICE" == "yes" || "$PROXY_USE_COMPOSE_SERVICE" == "on" ) ]]; then
  echo "[compose_up] Starting stack with proxy profile enabled"
  docker compose --profile proxy up --build -d
else
  echo "[compose_up] Starting stack without proxy profile"
  docker compose up --build -d
fi

echo "[compose_up] Services:"
docker compose ps
