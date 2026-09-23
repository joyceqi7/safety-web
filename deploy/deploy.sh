#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f .env ]; then
  cp .env.example .env
  echo 'Created .env. Set APP_PASSWORD and optional LLM_API_KEY, then rerun.'
  exit 1
fi
docker compose config --quiet
docker compose up -d --build --wait --wait-timeout 300
docker compose ps
