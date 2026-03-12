from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SAM2 체크포인트 설정 (facebook/sam2-hiera-large, ~900MB)
DEFAULT_SAM2_CHECKPOINT_PATH = PROJECT_ROOT / "pretrained_models" / "sam2.pt"
DEFAULT_SAM2_HF_REPO_ID = "facebook/sam2-hiera-large"
DEFAULT_SAM2_HF_FILENAME = "sam2_hiera_large.pt"

# SAM2 모델 config (sam2 패키지 내부 경로)
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"


def sam2_python_supported() -> bool:
    return True


def resolve_sam2_checkpoint_path(explicit_path: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    env_path = os.environ.get("SAM2_CHECKPOINT_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([
        DEFAULT_SAM2_CHECKPOINT_PATH,
        PROJECT_ROOT / "pretrained_models" / "sam2_hiera_large.pt",
    ])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def maybe_download_sam2_checkpoint(
    *,
    checkpoint_path: Optional[str] = None,
    auto_download: bool = True,
    hf_token: Optional[str] = None,
) -> Optional[Path]:
    resolved = resolve_sam2_checkpoint_path(checkpoint_path)
    if resolved is not None:
        return resolved
    if not auto_download:
        return None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    token = _resolve_hf_token(hf_token)
    target_path = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path
        else DEFAULT_SAM2_CHECKPOINT_PATH
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    repo_id = os.environ.get("SAM2_HF_REPO_ID", DEFAULT_SAM2_HF_REPO_ID)
    filename = os.environ.get("SAM2_HF_FILENAME", DEFAULT_SAM2_HF_FILENAME)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        local_dir=str(target_path.parent),
        local_dir_use_symlinks=False,
    )
    downloaded_path = Path(downloaded).resolve()
    if downloaded_path.name != target_path.name:
        target_path.write_bytes(downloaded_path.read_bytes())
        return target_path
    return downloaded_path


def create_sam2_predictor_factory(
    *,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
    auto_download: bool = True,
) -> Optional[Callable[[], Any]]:
    model_path = maybe_download_sam2_checkpoint(
        checkpoint_path=checkpoint_path,
        auto_download=auto_download,
    )
    if model_path is None:
        return None

    factory = _build_sam2_predictor_factory(device=device, checkpoint_path=model_path)
    return factory


def describe_missing_sam2_support(
    *,
    checkpoint_path: Optional[str] = None,
    auto_download: bool = True,
) -> Optional[str]:
    if not sam2_python_supported():
        return "SAM2 is enabled, but the installed runtime does not support this Python version."
    if resolve_sam2_checkpoint_path(checkpoint_path) is None and not auto_download:
        return "SAM2 is enabled, but no checkpoint was found. Set SAM2_CHECKPOINT_PATH or enable auto-download."
    try:
        import sam2  # noqa: F401
    except ImportError:
        return "SAM2 is enabled, but the sam2 Python package is not installed."
    return None


def _resolve_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    return (
        explicit_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("hf_token")
        or os.environ.get("huggingface_hub_token")
    )


def _build_sam2_predictor_factory(
    *, device: str, checkpoint_path: Path
) -> Optional[Callable[[], Any]]:
    """SAM2ImagePredictor factory — set_image/predict API 호환"""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        return None

    def _factory() -> Any:
        model_cfg = _resolve_sam2_config()
        sam2_model = build_sam2(model_cfg, str(checkpoint_path), device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        return predictor

    return _factory


def _resolve_sam2_config() -> str:
    """sam2 패키지 내부 config yaml 경로 반환"""
    env_cfg = os.environ.get("SAM2_MODEL_CONFIG")
    if env_cfg:
        return env_cfg

    # ⚠️  build_sam2()의 config_file 인자는 파일명만 받아야 함 (Hydra 제약)
    #     full path를 넘기면 MissingConfigException 발생
    try:
        import sam2
        pkg_dir = Path(sam2.__file__).parent
        for cfg_name in ("sam2_hiera_l.yaml", "sam2_hiera_large.yaml"):
            candidates = list(pkg_dir.rglob(cfg_name))
            if candidates:
                return candidates[0].name  # ← 파일명만 반환 (full path ❌)
    except Exception:
        pass

    return SAM2_MODEL_CONFIG
