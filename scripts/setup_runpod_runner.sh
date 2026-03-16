#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export GITHUB_OWNER="PracLee"
#   export GITHUB_REPO="hair_swap_model"
#   export GITHUB_RUNNER_PAT="<github_pat_or_classic_pat>"
#   export RUNNER_NAME="runpod-sd-builder-01"         # optional
#   export RUNNER_LABELS="runpod,sd-builder,linux,x64" # optional
#   bash scripts/setup_runpod_runner.sh

if [[ -z "${GITHUB_OWNER:-}" ]]; then
  echo "GITHUB_OWNER is required"
  exit 1
fi
if [[ -z "${GITHUB_REPO:-}" ]]; then
  echo "GITHUB_REPO is required"
  exit 1
fi
if [[ -z "${GITHUB_RUNNER_PAT:-}" ]]; then
  echo "GITHUB_RUNNER_PAT is required"
  exit 1
fi

RUNNER_VERSION="${RUNNER_VERSION:-2.325.0}"
RUNNER_HOME="${RUNNER_HOME:-/opt/actions-runner}"
RUNNER_NAME="${RUNNER_NAME:-runpod-$(hostname)-$(date +%s)}"
RUNNER_LABELS="${RUNNER_LABELS:-runpod,sd-builder,linux,x64}"
RUNNER_WORKDIR="${RUNNER_WORKDIR:-_work}"

echo "[runner] owner=${GITHUB_OWNER} repo=${GITHUB_REPO}"
echo "[runner] name=${RUNNER_NAME} labels=${RUNNER_LABELS}"

if [[ "$(id -u)" -eq 0 ]]; then
  # GitHub runner는 root 실행 시 명시적 opt-in이 필요함.
  export RUNNER_ALLOW_RUNASROOT=1
  echo "[runner] running as root -> RUNNER_ALLOW_RUNASROOT=1"
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends curl jq ca-certificates tar git docker.io
  rm -rf /var/lib/apt/lists/*
fi

# Start docker daemon when needed (common on RunPod container environments).
if ! docker info >/dev/null 2>&1; then
  # RunPod/컨테이너 환경에서는 overlay2 + namespace 권한 문제(unshare: operation not permitted)가
  # 자주 발생하므로 fallback 모드(vfs/no-iptables/no-bridge)를 기본값으로 사용한다.
  echo "[runner] docker daemon is not running. starting dockerd (fallback mode)..."
  pkill -f dockerd >/dev/null 2>&1 || true
  mkdir -p /tmp/docker-data /tmp/docker-exec
  nohup dockerd \
    --host=unix:///var/run/docker.sock \
    --data-root=/tmp/docker-data \
    --exec-root=/tmp/docker-exec \
    --storage-driver=vfs \
    --iptables=false \
    --ip-masq=false \
    --bridge=none \
    >/tmp/dockerd.log 2>&1 &

  for i in {1..45}; do
    if docker info >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! docker info >/dev/null 2>&1; then
  echo "[runner] docker daemon failed to start."
  echo "[runner] last 120 lines of /tmp/dockerd.log:"
  tail -n 120 /tmp/dockerd.log || true
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
