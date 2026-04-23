#!/usr/bin/env bash
# Run the full pytest suite inside Dockerfile.crewai (Python 3.11-slim), same stack as the app.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required." >&2
  exit 1
fi

exec docker build \
  -f "${REPO_ROOT}/Dockerfile.crewai" \
  --target tests \
  "${REPO_ROOT}"
