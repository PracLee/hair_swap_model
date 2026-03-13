#!/bin/bash
set -e

# ── 코드 최신화 (Docker 재빌드 없이 .py 변경 반영) ──────────────────────────
# RunPod 환경변수 GITHUB_TOKEN이 있으면 private repo도 pull 가능
echo "[entrypoint_sd] Pulling latest code from master..."
if [ -n "${GITHUB_TOKEN}" ]; then
    git -C /app pull "https://${GITHUB_TOKEN}@github.com/PracLee/hair_swap_model.git" master 2>&1 \
        && echo "[entrypoint_sd] git pull success" \
        || echo "[warn] git pull failed, using bundled code"
else
    git -C /app pull origin master 2>&1 \
        && echo "[entrypoint_sd] git pull success" \
        || echo "[warn] git pull failed (no GITHUB_TOKEN?), using bundled code"
fi

echo "[entrypoint_sd] Starting SD Inpainting handler..."
exec python handler_sd.py
