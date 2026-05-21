#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  echo "[start_gateway] .env not found in project root. Copy .env.example to .env first."
  exit 1
fi

python3 -m uvicorn unified_gateway.app.main:app --host "${UAG_APP_HOST:-0.0.0.0}" --port "${UAG_APP_PORT:-8080}"
