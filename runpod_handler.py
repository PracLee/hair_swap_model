from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

# ── 모듈 임포트 에러를 핸들러 레벨에서 반환하기 위해 지연 캡처 ──────────────
_IMPORT_ERROR: str | None = None
try:
    from PIL import Image, ImageOps
    from hairclip_assets import ensure_runtime_assets
    from pipeline_optimized import MirrAIOptimizedPipeline, ServiceConfig
    from utils.env_loader import load_project_dotenv
    from utils.runtime_compat import clear_cuda_memory, is_cuda_oom_error
    load_project_dotenv()
except Exception as _e:
    _IMPORT_ERROR = f"{type(_e).__name__}: {_e}\n{traceback.format_exc()}"
    # 더미 정의 (handler가 호출될 수 있도록)
    class MirrAIOptimizedPipeline: pass  # type: ignore
    class ServiceConfig: pass  # type: ignore
    def ensure_runtime_assets(*a, **kw): pass
    def load_project_dotenv(): pass
    def clear_cuda_memory(): pass
    def is_cuda_oom_error(e): return False


PROJECT_ROOT = Path(__file__).resolve().parent
RUNPOD_INPUT_DIR = PROJECT_ROOT / "output" / "runpod_inputs"
_PIPELINES: Dict[Tuple[str, str, str, bool, bool, int, int, float], MirrAIOptimizedPipeline] = {}
_PIPELINE_LOCK = threading.Lock()
DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_INPUT_PIXELS = 20_000_000
DEFAULT_RETURN_BASE64 = False
RUNPOD_INPUT_TTL_SECONDS = 6 * 60 * 60
RUNPOD_RESULT_TTL_SECONDS = 12 * 60 * 60


load_project_dotenv()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _download_image(url: str) -> Path:
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".png"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    target_path = RUNPOD_INPUT_DIR / f"url_{digest}{suffix}"
    if target_path.is_file():
        return target_path
    request = urllib.request.Request(url, headers={"User-Agent": "MirrAI-RunPod/1.0"})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"Remote image is too large: {content_length} bytes")
        remaining = MAX_DOWNLOAD_BYTES
        with target_path.open("wb") as handle:
            while True:
                chunk = response.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
                if remaining <= 0:
                    handle.close()
                    target_path.unlink(missing_ok=True)
                    raise ValueError(f"Remote image exceeds {MAX_DOWNLOAD_BYTES} bytes")
    return _normalize_input_image(target_path)


def _decode_base64_image(payload: str) -> Path:
    if payload.startswith("data:"):
        _, payload = payload.split(",", 1)
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("image_base64 is not valid base64 data.") from exc
    digest = hashlib.sha256(image_bytes).hexdigest()[:16]
    target_path = RUNPOD_INPUT_DIR / f"base64_{digest}.png"
    if not target_path.is_file():
        target_path.write_bytes(image_bytes)
    return _normalize_input_image(target_path)


def _resolve_image_path(payload: Dict[str, Any]) -> Path:
    RUNPOD_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = payload.get("image_path")
    image_url = payload.get("image_url")
    image_base64 = payload.get("image_base64")

    if image_path:
        resolved = Path(str(image_path)).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Input image not found: {resolved}")
        return resolved
    if image_url:
        return _download_image(str(image_url))
    if image_base64:
        return _decode_base64_image(str(image_base64))
    raise ValueError("One of image_path, image_url, or image_base64 is required.")


def _normalize_input_image(path: Path) -> Path:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if width * height > MAX_INPUT_PIXELS:
            scale = (MAX_INPUT_PIXELS / float(width * height)) ** 0.5
            resized = (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )
            image = image.resize(resized, Image.Resampling.LANCZOS)
        image.save(path, format="PNG")
    return path


def _cleanup_stale_outputs() -> None:
    now = time.time()
    _prune_older_than(RUNPOD_INPUT_DIR, RUNPOD_INPUT_TTL_SECONDS, now=now)
    _prune_older_than(PROJECT_ROOT / "output" / "optimized_results", RUNPOD_RESULT_TTL_SECONDS, now=now)


def _prune_older_than(root: Path, ttl_seconds: int, *, now: float) -> None:
    if ttl_seconds <= 0 or not root.exists():
        return
    for path in root.iterdir():
        try:
            age = now - path.stat().st_mtime
            if age <= ttl_seconds:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except Exception:
            continue


def _build_config(payload: Dict[str, Any]) -> ServiceConfig:
    return ServiceConfig(
        device=str(payload.get("device", "cuda")),
        preset=str(payload.get("preset", "balanced")),   # realtime → balanced (기본값 품질 향상)
        precision=str(payload.get("precision", "fp16")),
        enable_tensorrt=_coerce_bool(payload.get("enable_tensorrt")),
        enable_torch_compile=_coerce_bool(payload.get("enable_torch_compile")),
        async_load=not _coerce_bool(payload.get("disable_async_load")),
        trend_remote_url=payload.get("trend_remote_url"),
        trend_refresh_seconds=int(payload.get("trend_refresh_seconds", 1800)),
        trend_limit=int(payload.get("trend_limit", 5)),
        locale=str(payload.get("locale", "global")),
        alignment_intermediate_size=int(payload.get("alignment_intermediate_size", 2048)),
        alignment_upscale_min_size=int(payload.get("alignment_upscale_min_size", 1024)),
        alignment_max_upscale_factor=float(payload.get("alignment_max_upscale_factor", 2.0)),
        enable_sam2=_coerce_bool(payload.get("enable_sam2"), default=True),
        sam2_auto_download=not _coerce_bool(payload.get("disable_sam2_auto_download")),
        # 합성 품질 파라미터
        use_poisson_blend=_coerce_bool(payload.get("use_poisson_blend"), default=True),
        use_color_match=_coerce_bool(payload.get("use_color_match"), default=True),
        text_strength=float(payload.get("text_strength", 1.0)),
    )


def _build_pipeline_key(config: ServiceConfig) -> Tuple[str, str, str, bool, bool, int, int, float]:
    return (
        config.device,
        config.preset,
        config.precision,
        config.enable_tensorrt,
        config.enable_torch_compile,
        config.alignment_intermediate_size,
        config.alignment_upscale_min_size,
        float(config.alignment_max_upscale_factor),
    )


def _get_pipeline(config: ServiceConfig) -> MirrAIOptimizedPipeline:
    key = _build_pipeline_key(config)
    with _PIPELINE_LOCK:
        pipeline = _PIPELINES.get(key)
        if pipeline is None:
            pipeline = MirrAIOptimizedPipeline(config=config)
            _PIPELINES[key] = pipeline
        return pipeline


def _drop_pipeline(config: ServiceConfig) -> None:
    with _PIPELINE_LOCK:
        _PIPELINES.pop(_build_pipeline_key(config), None)


def _to_json_safe(obj: Any) -> Any:
    """numpy / torch / dataclass 등 JSON 비직렬화 타입을 재귀적으로 Python 기본형으로 변환."""
    # dict
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    # numpy scalar / array
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    # torch tensor
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    # dataclass → dict
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return _to_json_safe(dataclasses.asdict(obj))
    except Exception:
        pass
    # Path
    if isinstance(obj, Path):
        return str(obj)
    # 기본 JSON 지원 타입
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # 나머지: str 변환
    return str(obj)


def _cuda_info() -> dict:
    try:
        import torch
        return {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2) if torch.cuda.is_available() else None,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    # import 단계 실패 시 에러를 즉시 반환 (silent COMPLETED 방지)
    if _IMPORT_ERROR:
        # RunPod API: error 필드는 반드시 string 타입이어야 함
        return {
            "error": f"ImportError: Handler failed to import required modules at startup.\n{_IMPORT_ERROR}"
        }

    payload = dict((event or {}).get("input") or {})

    # 헬스체크: {"health_check": true} 로 인프라만 검증
    if _coerce_bool(payload.get("health_check")):
        return {
            "status": "ok",
            "python": sys.version,
            "cuda": _cuda_info(),
        }

    ensure_runtime_assets(restore_from_legacy=True)
    _cleanup_stale_outputs()

    config = _build_config(payload)
    try:
        pipeline = _get_pipeline(config)
        image_path = _resolve_image_path(payload)
        text_prompt = str(payload.get("text", ""))
        skip_align = _coerce_bool(payload.get("skip_align"))
        return_base64 = _coerce_bool(payload.get("return_base64"), default=DEFAULT_RETURN_BASE64)

        if _coerce_bool(payload.get("recommend_only")):
            analysis = pipeline.analyze_face(str(image_path))
            return {
                "face_analysis": analysis,
                "recommendation": pipeline.recommend_styles(analysis, prompt=text_prompt),
                "eta": pipeline.estimate_eta(str(image_path), text_prompt=text_prompt, skip_alignment=skip_align),
            }

        if _coerce_bool(payload.get("prepare_only")):
            result = pipeline.prepare_identity(str(image_path), skip_alignment=skip_align)
            result["eta"] = pipeline.estimate_eta(str(image_path), text_prompt=text_prompt, skip_alignment=skip_align)
            return result

        result = pipeline.generate_multiple_styles(
            image_path=str(image_path),
            text_prompt=text_prompt,
            skip_alignment=skip_align,
            return_base64=return_base64,
            output_dir=payload.get("output_dir"),
            limit=int(payload.get("trend_limit", config.trend_limit)),
        )
        import json as _json

        # ── 클라이언트에 필요한 필드만 추출 (face_analysis/performance 제거) ──
        results_safe = _to_json_safe(result.get("results", []))
        slim = {
            "styles": [
                {
                    "rank":        r.get("rank"),
                    "style_name":  r.get("style_name"),
                    "description": r.get("description"),
                    "score":       r.get("score"),
                    "image_base64": r.get("image_base64"),
                    # 진단 필드: 합성 여부, 출력 프레임, 마스킹 소스 확인용
                    "composited_to_original": r.get("composited_to_original"),
                    "output_frame":            r.get("output_frame"),
                    "masking":                 r.get("masking"),
                }
                for r in results_safe
            ],
            "eta": _to_json_safe(result.get("eta")),
        }

        # ── 직렬화 검증 + 크기 로깅 ──────────────────────────────────────────
        try:
            _raw = _json.dumps(slim)
            _kb = len(_raw) / 1024
            print(f"[handler] response size: {_kb:.1f} KB", flush=True)
        except Exception as _je:
            print(f"[handler] JSON serialization failed: {_je}", flush=True)
            # RunPod API: error 필드는 반드시 string 타입이어야 함
            return {"error": f"SerializationError: {_je}"}

        return slim
    except Exception as exc:
        if is_cuda_oom_error(exc):
            _drop_pipeline(config)
            clear_cuda_memory()
            # RunPod API: error 필드는 반드시 string 타입이어야 함
            return {
                "error": (
                    "cuda_out_of_memory: CUDA memory was exhausted. "
                    f"detail={exc} | "
                    "Tip: use preset=realtime, trend_limit<=3, return_base64=false, or smaller input image."
                )
            }
        # RunPod API: error 필드는 반드시 string 타입이어야 함
        import traceback as _tb
        return {"error": f"{exc.__class__.__name__}: {exc}\n{_tb.format_exc()}"}


def start_worker() -> None:
    import os
    import aiohttp as _aiohttp
    import runpod
    import runpod.serverless.modules.rp_http as _rp_http

    # ── 시작 진단 로그 ────────────────────────────────────────────────────────
    print(f"[startup] RUNPOD_POD_ID         = {os.environ.get('RUNPOD_POD_ID', 'NOT SET')}", flush=True)
    print(f"[startup] RUNPOD_WEBHOOK_POST_OUTPUT (raw) = {os.environ.get('RUNPOD_WEBHOOK_POST_OUTPUT', 'NOT SET')[:200]}", flush=True)
    print(f"[startup] JOB_DONE_URL (built)  = {_rp_http.JOB_DONE_URL}", flush=True)
    api_key = os.environ.get("RUNPOD_AI_API_KEY", "")
    print(f"[startup] RUNPOD_AI_API_KEY     = {'SET (len=' + str(len(api_key)) + ')' if api_key else 'NOT SET'}", flush=True)

    # ── _transmit monkey-patch ────────────────────────────────────────────────
    # 원인 분석:
    #   1. AsyncClientSession 기본 헤더: Content-Type: application/json
    #   2. 기존 _transmit이 이를 application/x-www-form-urlencoded 로 덮어씀
    #   3. RunPod API가 JSON Content-Type을 요구하면 400 발생
    #
    # 수정: application/json Content-Type 유지 + 응답 바디 로깅 추가
    async def _patched_transmit(client_session, url, job_data):
        print(f"[job-done] POST {url}", flush=True)
        print(f"[job-done] body_size={len(job_data)} preview={job_data[:120]}", flush=True)

        # Content-Type override 없이 session 기본값(application/json) 사용
        try:
            async with client_session.post(
                url,
                data=job_data,
                headers={"charset": "utf-8"},  # Content-Type은 session 기본값(json) 유지
                raise_for_status=False,
            ) as resp:
                body = await resp.text()
                print(f"[job-done] status={resp.status} response_body={body[:400]}", flush=True)
                if resp.status >= 400:
                    raise _aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        status=resp.status,
                        message=f"{resp.reason} | response: {body[:200]}",
                        headers=resp.headers,
                    )
        except _aiohttp.ClientError as exc:
            print(f"[job-done] ClientError: {exc}", flush=True)
            raise

    _rp_http._transmit = _patched_transmit

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    start_worker()
