"""
Dockerfile 빌드 시 실행되는 모델 다운로드 스크립트 (슬림화 버전).

빌드 시 포함 (cold start 병목 방지):
  - SD 1.5 Inpainting     ~4.0 GB  (매 요청 필수)
  - ControlNet Canny       ~1.5 GB  (매 요청 필수)
  - IP-Adapter weight      ~0.8 GB  (매 요청 필수)

런타임 자동 다운로드 (첫 cold start 1회만):
  - IP-Adapter image encoder ~2.4 GB  (diffusers load_ip_adapter 자동)
  - SAM2                     ~0.9 GB  (auto_download=True)
  - CLIP                     ~0.6 GB  (transformers 자동)
"""
import os
from huggingface_hub import hf_hub_download, snapshot_download

token = os.environ.get("HF_TOKEN") or None

# ── SD Inpainting ─────────────────────────────────────────────────────────────
print("[HF] Downloading SD Inpainting (runwayml/stable-diffusion-inpainting)...")
snapshot_download(
    "runwayml/stable-diffusion-inpainting",
    token=token,
    ignore_patterns=[
        "*.msgpack", "*.h5",
        "flax_model*", "tf_model*", "rust_model*",
        "*.onnx", "*.pb",
    ],
)
print("[HF] SD Inpainting OK")

# ── ControlNet Canny ──────────────────────────────────────────────────────────
print("[HF] Downloading ControlNet Canny (lllyasviel/control_v11p_sd15_canny)...")
snapshot_download(
    "lllyasviel/control_v11p_sd15_canny",
    token=token,
    ignore_patterns=["*.msgpack", "*.h5", "*.onnx"],
)
print("[HF] ControlNet OK")

# ── IP-Adapter weight only (image_encoder는 diffusers가 런타임에 자동 다운로드) ─
print("[HF] Downloading IP-Adapter face weight only...")
hf_hub_download(
    repo_id="h94/IP-Adapter",
    subfolder="models",
    filename="ip-adapter-plus-face_sd15.bin",
    token=token,
)
print("[HF] IP-Adapter weight OK")

# ── SAM2, CLIP, IP-Adapter image_encoder는 제외 ───────────────────────────────
#   SAM2       : pipeline_sd_inpainting.py auto_download=True로 첫 실행 시 받음
#   CLIP       : transformers 첫 호출 시 자동 다운로드
#   image_enc  : diffusers load_ip_adapter 내부에서 자동 다운로드

print("\n[download_weights_sd] 빌드 필수 모델 다운로드 완료 ✓")
print("  총 빌드 포함: SD(~4GB) + ControlNet(~1.5GB) + IP-Adapter weight(~0.8GB)")
print("  런타임 자동:  SAM2(~0.9GB) + CLIP(~0.6GB) + IP-Adapter encoder(~2.4GB)")
