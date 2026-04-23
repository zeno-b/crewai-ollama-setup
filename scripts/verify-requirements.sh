#!/usr/bin/env bash
# Resolve requirements.txt inside the same Python image stack as Dockerfile.crewai
# (no host venv, no GitHub Actions).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not on PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running or this user cannot access it." >&2
  exit 1
fi

exec docker build \
  -f "${REPO_ROOT}/Dockerfile.crewai" \
  --target requirements-verify \
  "${REPO_ROOT}"
