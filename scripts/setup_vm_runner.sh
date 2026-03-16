#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export GITHUB_OWNER="PracLee"
#   export GITHUB_REPO="hair_swap_model"
#   export GITHUB_RUNNER_PAT="<github_pat>"
#   export RUNNER_NAME="vm-sd-builder-01"              # optional
#   export RUNNER_LABELS="vm,sd-builder,linux,x64"     # optional
#   bash scripts/setup_vm_runner.sh

if [[ -z "${GITHUB_OWNER:-}" ]]; then
  echo "GITHUB_OWNER is required"
  exit 1
fi
if [[ -z "${GITHUB_REPO:-}" ]]; then
  echo "GITHUB_REPO is required"
  exit 1
fi

RUNNER_VERSION="${RUNNER_VERSION:-2.332.0}"
RUNNER_HOME="${RUNNER_HOME:-/opt/actions-runner}"
RUNNER_NAME="${RUNNER_NAME:-vm-$(hostname)-$(date +%s)}"
RUNNER_LABELS="${RUNNER_LABELS:-vm,sd-builder,linux,x64}"
RUNNER_WORKDIR="${RUNNER_WORKDIR:-_work}"

echo "[runner] owner=${GITHUB_OWNER} repo=${GITHUB_REPO}"
echo "[runner] name=${RUNNER_NAME} labels=${RUNNER_LABELS}"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends curl jq ca-certificates tar git docker.io
  rm -rf /var/lib/apt/lists/*
fi

if command -v systemctl >/dev/null 2>&1; then
  systemctl enable docker || true
  systemctl start docker || true
fi

if ! docker info >/dev/null 2>&1; then
  echo "[runner] docker daemon is not running on VM. start docker first."
  exit 1
fi

mkdir -p "${RUNNER_HOME}"
cd "${RUNNER_HOME}"

if [[ ! -f "./config.sh" ]]; then
  curl -fsSL -o actions-runner.tar.gz \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
  tar xzf actions-runner.tar.gz
  rm -f actions-runner.tar.gz
fi

# 이미 등록된 runner라면 재등록 없이 바로 실행
if [[ -f "./.runner" ]]; then
  echo "[runner] existing runner configuration detected. starting runner without reconfiguration..."
  exec ./run.sh
fi

if [[ -z "${GITHUB_RUNNER_PAT:-}" ]]; then
  echo "GITHUB_RUNNER_PAT is required for first-time runner registration"
  exit 1
fi

REG_TOKEN="$(
  curl -fsSL -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${GITHUB_RUNNER_PAT}" \
    "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/actions/runners/registration-token" \
  | jq -r '.token'
)"

if [[ -z "${REG_TOKEN}" || "${REG_TOKEN}" == "null" ]]; then
  echo "[runner] failed to get registration token"
  exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
  export RUNNER_ALLOW_RUNASROOT=1
fi

./config.sh \
  --url "https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}" \
  --token "${REG_TOKEN}" \
  --name "${RUNNER_NAME}" \
  --labels "${RUNNER_LABELS}" \
  --work "${RUNNER_WORKDIR}" \
  --replace \
  --unattended

echo "[runner] configured. starting runner..."
exec ./run.sh
