# =============================================================================
# Stage 1 – Python 의존성
#   • pytorch/pytorch 공식 베이스 (torch+cu128 포함)
#   • requirements 변경 시에만 이 레이어 재빌드
# =============================================================================
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel AS deps

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build build-essential git \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── requirements 파일만 먼저 복사 → 코드 변경 시 pip 캐시 재사용 ──────────
COPY requirements.txt requirements-sam3.txt ./

# ── Step A: 기본 패키지 설치 (torch/torchvision/opencv/ipython 제외) ────────
#   torch/torchvision: 베이스 이미지에 이미 포함
#   opencv: Step B에서 단독 처리 (mediapipe 충돌 방지)
RUN pip install --upgrade pip && \
    grep -vE "^torch|^torchvision|^--extra-index-url|^opencv|^ipython" \
        requirements.txt > /tmp/req_base.txt && \
    pip install -r /tmp/req_base.txt

# ── Step B: opencv-headless 강제 설치 ────────────────────────────────────────
#   mediapipe가 opencv를 자체 설치할 수 있으므로 headless로 덮어씀
RUN pip install --force-reinstall "opencv-python-headless==4.10.0.84"

# ── Step C: SAM2 의존성 설치 ──────────────────────────────────────────────────
#   hydra-core, iopath: sam2 런타임 필수 (requirements-sam3.txt에 명시됨)
#   sam2: git 설치 (pypi 미배포, --no-deps로 torch 중복 방지)
#   decord: 비디오 처리 전용 → 이미지 파이프라인 불필요, 제외
RUN grep -vE "^decord|^huggingface_hub|^git\+" requirements-sam3.txt \
        > /tmp/req_sam_deps.txt && \
    pip install -r /tmp/req_sam_deps.txt && \
    pip install --no-deps \
        "git+https://github.com/facebookresearch/sam2.git"


# =============================================================================
# Stage 2 – 런타임 이미지
#   • 모델: 로컬 파일 COPY (gdown 불필요 → 네트워크 장애 위험 0)
#   • SAM2 체크포인트만 HuggingFace에서 다운로드 (공개 모델, 인증 불필요)
#   • 소스코드는 마지막에 복사 → 코드 변경 시 모델 레이어 캐시 유지
# =============================================================================
FROM deps AS runtime

WORKDIR /app

# ── 사전 학습 모델 복사 (변경 빈도 낮음 → 캐시 고정 레이어) ─────────────────
#   .dockerignore에서 sam3.pt(3.2GB) 제외됨
#   ffhq.pt(127MB) + seg.pth(51MB) + bald_proxy.pt(8MB) + ffhq_PCA.npz(2MB)
COPY pretrained_models/ pretrained_models/

# ── SAM2 체크포인트 다운로드 (facebook/sam2-hiera-large, ~900MB) ─────────────
#   shutil.copy2 사용: read_bytes()의 OOM 위험 제거
#   local_dir 미사용: local_dir_use_symlinks deprecated 경고 없음
RUN python - << 'PYEOF'
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

target = Path("/app/pretrained_models/sam2.pt")
target.parent.mkdir(parents=True, exist_ok=True)

# HF 캐시에 다운로드 후 target 경로로 복사
cached = hf_hub_download(
    repo_id="facebook/sam2-hiera-large",
    filename="sam2_hiera_large.pt",
)
src = Path(cached).resolve()
shutil.copy2(src, target)
mb = target.stat().st_size // 1024 // 1024
print(f"[SAM2] ready: {target} ({mb} MB)")
PYEOF

# ── LPIPS 가중치 복사 (로컬에 존재, criteria/ COPY로 자동 포함됨) ────────────
#   criteria/lpips/weights/v0.1/vgg.pth 로컬에 있음 → 별도 다운로드 불필요

# ── 소스코드 복사 (자주 변경 → 마지막 레이어) ────────────────────────────────
COPY entrypoint.sh         ./
COPY runpod_handler.py     ./
COPY pipeline_optimized.py ./
COPY hairclip_assets.py    ./
COPY download_weights.py   ./
COPY runtime_spec.py       ./
COPY models/               models/
COPY scripts/              scripts/
COPY utils/                utils/
COPY criteria/             criteria/
COPY data/                 data/
COPY images/               images/

RUN chmod +x entrypoint.sh && \
    mkdir -p /app/output/torch_extensions

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_EXTENSIONS_DIR=/app/output/torch_extensions \
    ENABLE_SAM3=1

# ── StyleGAN2 CUDA 확장 빌드 시점 pre-compile ──────────────────────────────
#   콜드 스타트마다 JIT 컴파일(30~60 초)을 이미지 빌드 시점으로 이동.
#   RTX 4090(8.9) / RTX 5090(11.0) / A100(8.0) / A40(8.6) / H100(9.0) 멀티 아키텍처 대응.
#   nvcc(devel 이미지)가 CPU에서 컴파일 → GPU 없이도 빌드 가능.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;11.0+PTX" \
    HAIRCLIP_VERBOSE_EXTENSIONS=0

RUN python - << 'PYEOF'
import sys
sys.path.insert(0, '/app')
# import 시점에 load() 가 호출되어 .so 파일이 TORCH_EXTENSIONS_DIR에 저장됨
from models.stylegan2.op import upfirdn2d, fused_act   # noqa: F401
import os, pathlib
ext_dir = pathlib.Path(os.environ["TORCH_EXTENSIONS_DIR"])
built = list(ext_dir.rglob("*.so"))
print(f"[precompile] StyleGAN2 CUDA extensions ready: {[str(p) for p in built]}")
PYEOF

ENTRYPOINT ["./entrypoint.sh"]
