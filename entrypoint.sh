#!/bin/bash
set -e

# 레거시 백업 디렉토리에서 가중치 복원 시도 (없으면 no-op)
# --strict 제외: 모델은 이미 이미지에 포함됨, 시작 실패 방지
echo "[entrypoint] Restoring legacy assets (if any)..."
python download_weights.py --restore-legacy

echo "[entrypoint] Starting handler..."
exec python runpod_handler.py
