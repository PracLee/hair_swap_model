#!/bin/bash
set -e

echo "[entrypoint_sd] Starting SD Inpainting handler..."
exec python handler_sd.py
