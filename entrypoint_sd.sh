#!/bin/bash
set -e

ENABLE_STARTUP_GIT_PULL="${ENABLE_STARTUP_GIT_PULL:-0}"
GIT_PULL_REF="${GIT_PULL_REF:-master}"
ENABLE_STARTUP_GIT_PULL_LOWER="$(echo "${ENABLE_STARTUP_GIT_PULL}" | tr '[:upper:]' '[:lower:]')"

# 기본값은 불변 이미지 코드 사용 (재현성 우선).
# 필요 시 ENABLE_STARTUP_GIT_PULL=1|true|yes|on 으로만 git pull 수행.
if [ "${ENABLE_STARTUP_GIT_PULL_LOWER}" = "1" ] || \
   [ "${ENABLE_STARTUP_GIT_PULL_LOWER}" = "true" ] || \
   [ "${ENABLE_STARTUP_GIT_PULL_LOWER}" = "yes" ] || \
   [ "${ENABLE_STARTUP_GIT_PULL_LOWER}" = "on" ]; then
    echo "[entrypoint_sd] ENABLE_STARTUP_GIT_PULL=${ENABLE_STARTUP_GIT_PULL} -> pulling ${GIT_PULL_REF}"
    if [ ! -d /app/.git ]; then
        echo "[entrypoint_sd] /app/.git not found. bootstrap git metadata..."
        git -C /app init
        git -C /app config --global --add safe.directory /app
        git -C /app remote add origin https://github.com/PracLee/hair_swap_model.git || true
    fi

    if [ -n "${GITHUB_TOKEN}" ]; then
        git -C /app fetch --depth 1 "https://${GITHUB_TOKEN}@github.com/PracLee/hair_swap_model.git" "${GIT_PULL_REF}" 2>&1 \
            && git -C /app reset --hard FETCH_HEAD 2>&1 \
            && echo "[entrypoint_sd] git pull success" \
            || echo "[warn] git pull failed, using bundled code"
    else
        git -C /app fetch --depth 1 origin "${GIT_PULL_REF}" 2>&1 \
            && git -C /app reset --hard FETCH_HEAD 2>&1 \
            && echo "[entrypoint_sd] git pull success" \
            || echo "[warn] git pull failed (no GITHUB_TOKEN?), using bundled code"
    fi
else
    echo "[entrypoint_sd] ENABLE_STARTUP_GIT_PULL=${ENABLE_STARTUP_GIT_PULL} -> using bundled image code"
fi

echo "[entrypoint_sd] Starting SD Inpainting handler..."
exec python handler_sd.py
