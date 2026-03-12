# backward-compatibility shim — sam3 → sam2 마이그레이션 완료
# 신규 코드는 utils/sam2_runtime.py 를 직접 임포트하세요.
from utils.sam2_runtime import (
    SAM2_MODEL_CONFIG,
    DEFAULT_SAM2_CHECKPOINT_PATH as DEFAULT_SAM3_CHECKPOINT_PATH,
    DEFAULT_SAM2_HF_REPO_ID as DEFAULT_SAM3_HF_REPO_ID,
    DEFAULT_SAM2_HF_FILENAME as DEFAULT_SAM3_HF_FILENAME,
    sam2_python_supported as sam3_python_supported,
    resolve_sam2_checkpoint_path as resolve_sam3_checkpoint_path,
    maybe_download_sam2_checkpoint as maybe_download_sam3_checkpoint,
    create_sam2_predictor_factory as create_sam3_predictor_factory,
    describe_missing_sam2_support as describe_missing_sam3_support,
)

__all__ = [
    "SAM2_MODEL_CONFIG",
    "DEFAULT_SAM3_CHECKPOINT_PATH",
    "DEFAULT_SAM3_HF_REPO_ID",
    "DEFAULT_SAM3_HF_FILENAME",
    "sam3_python_supported",
    "resolve_sam3_checkpoint_path",
    "maybe_download_sam3_checkpoint",
    "create_sam3_predictor_factory",
    "describe_missing_sam3_support",
]
