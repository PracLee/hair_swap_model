# =============================================================================
# MirrAI SD App Image (Code only)
# byoungj/sd:v*
#
# BASE_IMAGE는 CI에서 주입:
#   --build-arg BASE_IMAGE=<DOCKER_USERNAME>/sd:base-latest
# =============================================================================

ARG BASE_IMAGE=sd:base-latest
FROM ${BASE_IMAGE} AS runtime

WORKDIR /app

# SD 앱 코드만 복사
COPY --chmod=755 entrypoint_sd.sh entrypoint.sh
COPY handler_sd.py             ./
COPY pipeline_sd_inpainting.py ./
COPY runtime_download.py       ./
COPY utils/sam2_runtime.py     utils/sam2_runtime.py
COPY models/__init__.py        models/__init__.py
COPY models/face_parsing/      models/face_parsing/
COPY models/segface/           models/segface/
COPY data/                     data/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    SEG_PTH_GITHUB_OWNER=PracLee \
    SEG_PTH_GITHUB_REPO=hair_swap_model \
    SEG_PTH_GITHUB_REF=master \
    SEG_PTH_GITHUB_PATH=pretrained_models/seg.pth \
    MODEL_DOWNLOAD_TIMEOUT=600 \
    ENABLE_SAM2=1 \
    ENABLE_ROI_STRONGER_INPAINTER=1 \
    ROI_STRONGER_BACKEND=sdxl \
    ROI_STRONGER_CONTROL_MODE=canny \
    ROI_STRONGER_TARGET_SIZE=512 \
    ROI_STRONGER_STEPS=12 \
    ROI_STRONGER_GUIDANCE_SCALE=5.5 \
    ROI_STRONGER_CONDITIONING_SCALE=0.22 \
    ROI_STRONGER_STRENGTH=0.48 \
    ROI_STRONGER_MASK_EXPAND_PX=4 \
    ROI_STRONGER_MASK_BLUR_PX=4 \
    ENABLE_STARTUP_GIT_PULL=0 \
    GIT_PULL_REF=master

ENTRYPOINT ["python", "/app/handler_sd.py"]
