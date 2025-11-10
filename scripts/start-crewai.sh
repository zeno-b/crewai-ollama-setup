#!/usr/bin/env bash
set -euo pipefail

PORT="${CREWAI_PORT:-8000}"
DEFAULT_TIMEOUT="${GUNICORN_TIMEOUT:-90}"
DEFAULT_GRACEFUL="${GUNICORN_GRACEFUL_TIMEOUT:-30}"

if [[ -z "${WORKERS:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    CPU_COUNT="$(nproc)"
  else
    CPU_COUNT="2"
  fi
  # Gunicorn recommendation: 2 * $num_cores + 1
  WORKERS=$(( CPU_COUNT * 2 + 1 ))
fi

exec gunicorn \
  main:app \
  --bind "0.0.0.0:${PORT}" \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers "${WORKERS}" \
  --timeout "${DEFAULT_TIMEOUT}" \
  --graceful-timeout "${DEFAULT_GRACEFUL}" \
  --worker-tmp-dir /dev/shm \
  ${GUNICORN_EXTRA:-}
