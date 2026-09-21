#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '\n==> %s\n' "$*"
}

if [[ "$(uname -s)" != "Linux" ]]; then
  die "Run this script inside Linux/WSL2, not directly in PowerShell."
fi

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
fi

docker_cmd() {
  "${DOCKER[@]}" "$@"
}

COMPOSE=("${DOCKER[@]}" compose -f compose.yaml -f compose.gpu.yaml)

info "Checking host GPU access"
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  if [[ ! -e /dev/dxg ]]; then
    die "WSL2 cannot see /dev/dxg. Update the Windows NVIDIA driver, run 'wsl --update' and 'wsl --shutdown' in PowerShell, then retry."
  fi
  NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
else
  NVIDIA_SMI="$(command -v nvidia-smi || true)"
fi

[[ -n "$NVIDIA_SMI" && -x "$NVIDIA_SMI" ]] || die "nvidia-smi is unavailable. Install/update the host NVIDIA driver first. Do not install a Linux NVIDIA driver inside WSL2."
"$NVIDIA_SMI" || die "The host NVIDIA driver cannot access the GPU. Fix the host driver before configuring Docker."

info "Checking Docker daemon"
docker_cmd info >/dev/null || die "Cannot access Docker daemon. Use sudo or add the current user to the docker group."

if ! docker_cmd info --format '{{json .Runtimes}}' | grep -q 'nvidia'; then
  if ! systemctl is-active --quiet docker 2>/dev/null; then
    die "Docker does not expose an NVIDIA runtime. If this is Docker Desktop, enable WSL2 GPU support, update Docker Desktop, and restart it."
  fi

  info "Installing NVIDIA Container Toolkit for the WSL Docker Engine"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

info "Testing NVIDIA runtime"
docker_cmd info --format '{{json .Runtimes}}' | grep -q 'nvidia' \
  || die "Docker still has no NVIDIA runtime after configuration. Check: sudo journalctl -u docker -n 100 --no-pager"
docker_cmd run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

info "Checking the project image"
docker_cmd image inspect local-lora-pipeline:0.1.0 >/dev/null \
  || die "local-lora-pipeline:0.1.0 is missing. Build it first with: sudo docker compose -f compose.yaml -f compose.gpu.yaml build --progress=plain"

info "Running project preflight"
"${COMPOSE[@]}" run --rm --no-deps worker lora-pipeline preflight

printf '\nGPU Docker preflight completed successfully.\n'
