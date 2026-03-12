#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

HF_TOKEN="${HF_TOKEN:-${hf_token:-}}"
HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${huggingface_hub_token:-}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_DEV_EXTRAS="${INSTALL_DEV_EXTRAS:-0}"
INSTALL_SAM2="${INSTALL_SAM2:-0}"
DOWNLOAD_MISSING_WEIGHTS="${DOWNLOAD_MISSING_WEIGHTS:-1}"
SAM2_CHECKPOINT_PATH="${SAM2_CHECKPOINT_PATH:-$SCRIPT_DIR/pretrained_models/sam2.pt}"

echo "[1/6] Installing runtime Python requirements"
"$PYTHON_BIN" -m pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[2/6] Verifying Ninja build dependency"
if ! command -v ninja >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install ninja
fi
ninja --version >/dev/null

if [[ "$INSTALL_SAM2" == "1" ]]; then
  echo "[3/6] Installing optional SAM2 dependencies"
  "$PYTHON_BIN" -m pip install -r requirements-sam3.txt
else
  echo "[3/6] Skipping optional SAM2 dependencies"
fi

if [[ "$INSTALL_DEV_EXTRAS" == "1" ]]; then
  echo "[4/6] Installing optional notebook dependencies"
  "$PYTHON_BIN" -m pip install -r requirements-dev.txt
else
  echo "[4/6] Skipping optional notebook dependencies"
fi

ASSET_ARGS=(--restore-legacy --repair-corrupt --strict)
if [[ "$INSTALL_SAM2" == "1" ]]; then
  if [[ -n "${SAM2_CHECKPOINT_PATH:-}" ]]; then
    ASSET_ARGS+=(--sam2-checkpoint "$SAM2_CHECKPOINT_PATH")
  fi
  if [[ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" || -n "${SAM2_CHECKPOINT_PATH:-}" ]]; then
    ASSET_ARGS+=(--download-sam2)
  else
    echo "[5/6] SAM2 package installed, but checkpoint download is skipped because HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set."
  fi
fi

if [[ "$DOWNLOAD_MISSING_WEIGHTS" == "1" ]]; then
  echo "[5/6] Restoring, verifying, and downloading runtime weights"
  ASSET_ARGS+=(--download-missing)
else
  echo "[5/6] Restoring and verifying runtime weights without network downloads"
fi
"$PYTHON_BIN" download_weights.py "${ASSET_ARGS[@]}"

echo "[6/6] Runtime bootstrap complete"
