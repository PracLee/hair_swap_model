"""
runtime_download.py - 런팟 cold start 시 모델 다운로드
빌드에 모델 포함 안 함 → Docker 이미지 크기 ~6GB 절감
handler_sd.py에서 이 함수를 호출함
"""
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def ensure_models_cached() -> None:
    """
    Cold start 시 HF Hub에서 모델 다운로드.
    이미 캐시됐으면 즉시 리턴 (RunPod Network Volume 마운트 시 재사용).
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    token = os.environ.get("HF_TOKEN") or None

    models = [
        ("SD Inpainting",   "runwayml/stable-diffusion-inpainting",  None, None,
         ["*.msgpack","*.h5","flax_model*","tf_model*","rust_model*"]),
        ("ControlNet Canny","lllyasviel/control_v11p_sd15_canny",     None, None,
         ["*.msgpack","*.h5"]),
    ]

    for name, repo_id, subfolder, filename, ignore in models:
        logger.info(f"[models] {name} 확인 중...")
        try:
            snapshot_download(
                repo_id, token=token,
                ignore_patterns=ignore or [],
                local_files_only=True,   # 이미 있으면 바로 OK
            )
            logger.info(f"[models] {name} 캐시 확인 ✓")
        except Exception:
            logger.info(f"[models] {name} 다운로드 중...")
            snapshot_download(repo_id, token=token, ignore_patterns=ignore or [])
            logger.info(f"[models] {name} 완료 ✓")

    # IP-Adapter weight
    _ensure_file("IP-Adapter weight", "h94/IP-Adapter",
                 "ip-adapter-plus-face_sd15.bin", "models", token)


def _ensure_file(name, repo_id, filename, subfolder, token):
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(repo_id, filename=filename,
                        subfolder=subfolder, token=token,
                        local_files_only=True)
        logger.info(f"[models] {name} 캐시 확인 ✓")
    except Exception:
        logger.info(f"[models] {name} 다운로드 중...")
        hf_hub_download(repo_id, filename=filename,
                        subfolder=subfolder, token=token)
        logger.info(f"[models] {name} 완료 ✓")
