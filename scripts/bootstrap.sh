#!/usr/bin/env bash

set -euo pipefail

if [[ "${BOOTSTRAP_DEBUG:-0}" == "1" ]]; then
  set -x
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUDO=""
RUN_USER="${SUDO_USER:-$(whoami)}"

if [[ $EUID -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "This script requires administrative privileges. Please re-run as root or install sudo." >&2
    exit 1
  fi
fi

NON_INTERACTIVE=0
INSTALL_OLLAMA_CLI=0
VERIFY_REQUIREMENTS=0
OLLAMA_MODEL="${OLLAMA_MODEL:-llama2:7b}"

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [options]

Ensure all prerequisites (Docker, Docker Compose, Ollama) are installed and basic
runtime directories are prepared. By default the script runs interactively.

Options:
  --non-interactive        Run without interactive prompts (assumes "yes").
  --with-ollama-cli        Install the Ollama CLI on the host in addition to Docker service.
  --verify-requirements    After Docker is available, run pip install -r requirements.txt inside
                           the same base image as Dockerfile.crewai (catches pin conflicts early).
  --ollama-model <name>    Default model to pre-pull inside the Ollama container (default: llama2:7b).
  -h, --help               Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --non-interactive)
      NON_INTERACTIVE=1
      shift
      ;;
    --with-ollama-cli)
      INSTALL_OLLAMA_CLI=1
      shift
      ;;
    --verify-requirements)
      VERIFY_REQUIREMENTS=1
      shift
      ;;
    --ollama-model)
      if [[ $# -lt 2 ]]; then
        echo "Error: --ollama-model requires a value." >&2
        exit 1
      fi
      OLLAMA_MODEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

prompt_confirm() {
  local message="${1:-Proceed?}"
  local default="${2:-y}"

  if [[ $NON_INTERACTIVE -eq 1 ]]; then
    [[ "${default,,}" == "y" ]]
    return
  fi

  local prompt="[Y/n]"
  if [[ "${default,,}" == "n" ]]; then
    prompt="[y/N]"
  fi

  while true; do
    read -r -p "$message $prompt " response
    response="${response:-$default}"
    case "${response,,}" in
      y|yes) return 0 ;;
      n|no) return 1 ;;
      *) echo "Please answer yes or no." ;;
    esac
  done
}

detected_os_id=""
detected_os_codename=""

detect_os() {
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    detected_os_id="${ID:-unknown}"
    detected_os_codename="${VERSION_CODENAME:-}"
  else
    detected_os_id="unknown"
    detected_os_codename=""
  fi
}

ensure_docker() {
  if command_exists docker; then
    echo "Docker already installed."
    return 0
  fi

  detect_os
  case "$detected_os_id" in
    ubuntu|debian)
      echo "Installing Docker Engine via apt for ${detected_os_id}..."
      $SUDO apt-get update
      $SUDO apt-get install -y ca-certificates curl gnupg lsb-release
      $SUDO install -m 0755 -d /etc/apt/keyrings
      curl -fsSL "https://download.docker.com/linux/${detected_os_id}/gpg" | $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      $SUDO chmod a+r /etc/apt/keyrings/docker.gpg

      local arch
      arch="$(dpkg --print-architecture)"
      local codename="${detected_os_codename:-$(. /etc/os-release; echo "${VERSION_CODENAME}")}"
      echo \
"deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${detected_os_id} ${codename} stable" | \
        $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null

      $SUDO apt-get update
      $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
      if command -v systemctl >/dev/null 2>&1; then
        $SUDO systemctl enable --now docker
      fi
      ;;
    *)
      cat <<EOF
Unsupported or undetected OS: ${detected_os_id}

Please install Docker manually by following the official instructions:
  https://docs.docker.com/engine/install/
EOF
      return 1
      ;;
  esac

  echo "Docker installation complete."
}

ensure_docker_running() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  echo "Attempting to start Docker service..."
  if command_exists systemctl; then
    $SUDO systemctl start docker || true
  fi

  if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not running. Please start Docker and re-run this script." >&2
    exit 1
  fi
}

ensure_docker_group() {
  if ! getent group docker >/dev/null 2>&1; then
    $SUDO groupadd docker
  fi

  if id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx "docker"; then
    return 0
  fi

  if prompt_confirm "Add ${RUN_USER} to docker group?" "y"; then
    $SUDO usermod -aG docker "$RUN_USER"
    echo "Added ${RUN_USER} to docker group. You must log out/in for changes to take effect."
  else
    echo "Skipping docker group membership update."
  fi
}

ensure_docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    echo "Docker Compose plugin detected."
    return 0
  fi

  if command_exists docker-compose; then
    echo "Legacy docker-compose detected."
    return 0
  fi

  echo "Docker Compose not found; installing plugin..."
  detect_os
  case "$detected_os_id" in
    ubuntu|debian)
      $SUDO apt-get update
      $SUDO apt-get install -y docker-compose-plugin
      ;;
    *)
      cat <<EOF
Unable to install Docker Compose automatically on ${detected_os_id}.
Follow the official instructions:
  https://docs.docker.com/compose/install/
EOF
      return 1
      ;;
  esac
}

ensure_directories() {
  mkdir -p \
    "${REPO_ROOT}/data/ollama" \
    "${REPO_ROOT}/data/crewai" \
    "${REPO_ROOT}/data/datasets" \
    "${REPO_ROOT}/data/retraining/jobs" \
    "${REPO_ROOT}/logs" \
    "${REPO_ROOT}/models" \
    "${REPO_ROOT}/config/grafana/provisioning" \
    "${REPO_ROOT}/config/prometheus"

  $SUDO chown -R "${RUN_USER}":"${RUN_USER}" \
    "${REPO_ROOT}/data" \
    "${REPO_ROOT}/logs" \
    "${REPO_ROOT}/models"
}

compose_cmd=()

resolve_compose_command() {
  if docker compose version >/dev/null 2>&1; then
    compose_cmd=(docker compose)
  elif command_exists docker-compose; then
    compose_cmd=(docker-compose)
  else
    echo "Docker Compose command not available." >&2
    return 1
  fi
}

ensure_ollama_cli() {
  if command_exists ollama; then
    echo "Ollama CLI already installed."
    return 0
  fi

  if [[ $INSTALL_OLLAMA_CLI -eq 0 ]]; then
    echo "Ollama CLI not installed. Skipping (use --with-ollama-cli to install)."
    return 0
  fi

  if ! prompt_confirm "Install Ollama CLI on host using official installer?" "y"; then
    echo "Skipping Ollama CLI installation."
    return 0
  fi

  local installer="/tmp/ollama-install.sh"
  curl -fsSL https://ollama.ai/install.sh -o "$installer"
  $SUDO chmod +x "$installer"
  $SUDO "$installer"
  rm -f "$installer"
  echo "Ollama CLI installation complete."
}

verify_requirements_in_container() {
  echo "Verifying requirements.txt resolves (Dockerfile.crewai target requirements-verify)..."
  docker build -f "${REPO_ROOT}/Dockerfile.crewai" --target requirements-verify "${REPO_ROOT}"
}

ensure_compose_services() {
  resolve_compose_command

  if [[ $VERIFY_REQUIREMENTS -eq 1 ]]; then
    verify_requirements_in_container
  fi

  echo "Pulling required Docker images (Ollama only for now)..."
  "${compose_cmd[@]}" -f "${REPO_ROOT}/docker-compose.yml" pull ollama

  echo "Starting Ollama service..."
  "${compose_cmd[@]}" -f "${REPO_ROOT}/docker-compose.yml" up -d ollama

  echo "Waiting for Ollama health check..."
  local retries=20
  until "${compose_cmd[@]}" -f "${REPO_ROOT}/docker-compose.yml" ps --services --filter "status=running" | grep -qx "ollama"; do
    sleep 3
    retries=$((retries - 1))
    if (( retries == 0 )); then
      echo "Ollama container did not become healthy in time. Check logs with: ${compose_cmd[*]} logs ollama" >&2
      return 1
    fi
  done

  echo "Ensuring default model (${OLLAMA_MODEL}) is available..."
  if command_exists ollama; then
    ollama list | grep -q "$OLLAMA_MODEL" || ollama pull "$OLLAMA_MODEL"
  else
    "${compose_cmd[@]}" -f "${REPO_ROOT}/docker-compose.yml" exec ollama ollama pull "$OLLAMA_MODEL"
  fi
}

main() {
  ensure_docker
  ensure_docker_running
  ensure_docker_group
  ensure_docker_compose
  ensure_directories
  ensure_ollama_cli
  ensure_compose_services

  cat <<EOF

Prerequisite setup complete.

Next steps:
  1. If you were added to the docker group, log out and log back in.
  2. Start the full stack when ready:
       ${compose_cmd[*]:-docker compose} -f docker-compose.yml up -d
  3. Access CrewAI at http://localhost:8000 once services are running.
EOF
}

main "$@"
