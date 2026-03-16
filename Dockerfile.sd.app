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

# BiSeNet 가중치만 로컬 포함 (상대적으로 작음)
COPY pretrained_models/seg.pth pretrained_models/seg.pth

# SD 앱 코드만 복사
COPY entrypoint_sd.sh          entrypoint.sh
COPY handler_sd.py             ./
COPY pipeline_sd_inpainting.py ./
COPY runtime_download.py       ./
COPY utils/sam2_runtime.py     utils/sam2_runtime.py
COPY models/__init__.py        models/__init__.py
COPY models/face_parsing/      models/face_parsing/
COPY models/segface/           models/segface/
COPY data/                     data/

# 선택적 startup git pull을 위한 최소 git metadata
RUN git init && \
    git remote add origin https://github.com/PracLee/hair_swap_model.git && \
    git config --global --add safe.directory /app

RUN chmod +x entrypoint.sh && \
    mkdir -p /app/output/runpod_inputs

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    ENABLE_SAM2=1 \
    ENABLE_STARTUP_GIT_PULL=0 \
    GIT_PULL_REF=master

ENTRYPOINT ["./entrypoint.sh"]
