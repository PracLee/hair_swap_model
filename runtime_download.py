"""
runtime_download.py - 런팟 cold start 시 모델 다운로드
빌드에 모델 포함 안 함 → Docker 이미지 크기 ~6GB 절감
handler_sd.py에서 이 함수를 호출함
"""
import logging
import os
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
PROJECT_ROOT = Path(__file__).resolve().parent
SEG_LOCAL_PATH = PROJECT_ROOT / "pretrained_models" / "seg.pth"


def ensure_models_cached() -> None:
    """
    Cold start 시 HF Hub에서 모델 다운로드.
    이미 캐시됐으면 즉시 리턴 (RunPod Network Volume 마운트 시 재사용).
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    token = os.environ.get("HF_TOKEN") or None

    # BiSeNet checkpoint: 이미지 포함 대신 런타임 다운로드
    _ensure_seg_checkpoint()

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


def _ensure_seg_checkpoint() -> None:
    seg_path = Path(os.environ.get("SEG_PTH_PATH", str(SEG_LOCAL_PATH)))
    seg_url = (os.environ.get("SEG_PTH_URL") or "").strip()
    timeout_sec = int(os.environ.get("MODEL_DOWNLOAD_TIMEOUT", "600"))
    min_bytes = int(os.environ.get("SEG_PTH_MIN_BYTES", str(10 * 1024 * 1024)))
    github_token = os.environ.get("GITHUB_TOKEN") or ""
    github_owner = os.environ.get("SEG_PTH_GITHUB_OWNER", "PracLee")
    github_repo = os.environ.get("SEG_PTH_GITHUB_REPO", "hair_swap_model")
    github_ref = os.environ.get("SEG_PTH_GITHUB_REF", "master")
    github_path = os.environ.get("SEG_PTH_GITHUB_PATH", "pretrained_models/seg.pth")

    if seg_path.exists() and seg_path.stat().st_size >= min_bytes:
        logger.info(f"[models] BiSeNet seg.pth 캐시 확인 ✓ ({seg_path})")
        return

    seg_path.parent.mkdir(parents=True, exist_ok=True)
    if seg_url:
        logger.info(f"[models] BiSeNet seg.pth 다운로드 중... ({seg_url})")
        _download_binary(seg_url, seg_path, timeout_sec=timeout_sec)
    elif github_token:
        logger.info(
            "[models] BiSeNet seg.pth 다운로드 중... "
            f"(github api: {github_owner}/{github_repo}@{github_ref}:{github_path})"
        )
        _download_from_github_api(
            dst_path=seg_path,
            token=github_token,
            owner=github_owner,
            repo=github_repo,
            ref=github_ref,
            file_path=github_path,
            timeout_sec=timeout_sec,
        )
    else:
        raise RuntimeError(
            "seg.pth not found. Set SEG_PTH_URL, or provide GITHUB_TOKEN "
            "(with optional SEG_PTH_GITHUB_OWNER/REPO/REF/PATH), "
            "or mount pretrained_models/seg.pth."
        )

    size = seg_path.stat().st_size if seg_path.exists() else 0
    if size < min_bytes:
        raise RuntimeError(f"seg.pth 다운로드 결과가 비정상입니다. size={size}")
    logger.info(f"[models] BiSeNet seg.pth 완료 ✓ ({size} bytes)")


def _download_binary(url: str, dst_path: Path, timeout_sec: int = 600) -> None:
    headers = {"User-Agent": "MirrAI-SD/1.0"}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token and ("github.com" in url or "githubusercontent.com" in url):
        headers["Authorization"] = f"Bearer {github_token}"

    req = urllib.request.Request(url, headers=headers)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp, open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp_path.replace(dst_path)
    except urllib.error.HTTPError as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"seg.pth HTTP 다운로드 실패 ({e.code}): {url}") from e
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _download_from_github_api(
    dst_path: Path,
    token: str,
    owner: str,
    repo: str,
    ref: str,
    file_path: str,
    timeout_sec: int = 600,
) -> None:
    quoted_path = urllib.parse.quote(file_path.lstrip("/"))
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{quoted_path}?ref={urllib.parse.quote(ref)}"
    headers = {
        "User-Agent": "MirrAI-SD/1.0",
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw",
    }

    req = urllib.request.Request(url, headers=headers)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp, open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp_path.replace(dst_path)
    except urllib.error.HTTPError as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"seg.pth GitHub API 다운로드 실패 ({e.code}): "
            f"{owner}/{repo}@{ref}:{file_path}"
        ) from e
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
