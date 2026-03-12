from __future__ import annotations

import importlib
import os
import types
import urllib.request
from pathlib import Path
from typing import Any, Optional


def safe_torch_load(path: Any, *, map_location: Any = "cpu", weights_only: bool = True, **kwargs: Any) -> Any:
    import torch

    load_kwargs = dict(kwargs)
    if map_location is not None:
        load_kwargs["map_location"] = map_location

    if "weights_only" not in load_kwargs:
        load_kwargs["weights_only"] = weights_only

    try:
        return torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        return torch.load(path, **load_kwargs)


def configure_torch_cuda_arch_list(device: Optional[Any] = None) -> Optional[str]:
    import torch

    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    if not torch.cuda.is_available():
        return None

    device_index = _resolve_cuda_device_index(device)
    major, minor = torch.cuda.get_device_capability(device_index)
    arch = f"{major}.{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    return arch


def configure_torch_extension_build_env(project_root: Any, *, default_max_jobs: int = 4) -> Path:
    project_root = Path(project_root)
    extensions_dir = project_root / "output" / "torch_extensions"
    extensions_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(extensions_dir))
    if default_max_jobs > 0:
        cpu_count = os.cpu_count() or default_max_jobs
        os.environ.setdefault("MAX_JOBS", str(max(1, min(default_max_jobs, cpu_count))))
    return extensions_dir


def build_mediapipe_face_mesh(**kwargs: Any) -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is not installed.") from exc

    face_mesh_module = getattr(getattr(mp, "solutions", None), "face_mesh", None)
    if face_mesh_module is None:
        for module_name in ("mediapipe.solutions.face_mesh", "mediapipe.python.solutions.face_mesh"):
            try:
                face_mesh_module = importlib.import_module(module_name)
                break
            except Exception:
                continue
    if face_mesh_module is not None:
        return face_mesh_module.FaceMesh(**kwargs)

    # Newer mediapipe builds may only expose the Tasks API.
    try:
        return _build_tasks_face_mesh_adapter(mp=mp, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "mediapipe FaceMesh could not be imported. "
            "This environment only exposes MediaPipe Tasks. "
            "Set MEDIAPIPE_FACE_LANDMARKER_MODEL to a valid face_landmarker.task model file "
            "or install a mediapipe build that includes solutions.face_mesh."
        ) from exc


def _build_tasks_face_mesh_adapter(mp: Any, **kwargs: Any) -> Any:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import face_landmarker

    model_path = _resolve_face_landmarker_model_path(mp)
    options = face_landmarker.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=_resolve_face_landmarker_running_mode(kwargs.get("static_image_mode", True)),
        num_faces=int(kwargs.get("max_num_faces", 1)),
        min_face_detection_confidence=float(kwargs.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(kwargs.get("min_tracking_confidence", 0.5)),
    )
    landmarker = face_landmarker.FaceLandmarker.create_from_options(options)
    return _MediaPipeTasksFaceMeshAdapter(mp=mp, landmarker=landmarker)


def _resolve_face_landmarker_running_mode(static_image_mode: Any) -> Any:
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

    return VisionTaskRunningMode.IMAGE if bool(static_image_mode) else VisionTaskRunningMode.VIDEO


def _resolve_face_landmarker_model_path(mp: Any) -> Path:
    env_model_path = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    if env_model_path:
        path = Path(env_model_path).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(
            f"MEDIAPIPE_FACE_LANDMARKER_MODEL is set but the file does not exist: {path}"
        )

    candidate_paths = [
        Path(mp.__file__).resolve().parent / "modules" / "face_landmarker" / "face_landmarker.task",
        Path(mp.__file__).resolve().parent / "tasks" / "testdata" / "vision" / "face_landmarker.task",
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate

    downloaded = _download_face_landmarker_model_if_allowed()
    if downloaded is not None:
        return downloaded

    raise FileNotFoundError(
        "No face_landmarker.task model found. "
        "Set MEDIAPIPE_FACE_LANDMARKER_MODEL to the local model path."
    )


def _download_face_landmarker_model_if_allowed() -> Optional[Path]:
    auto_download = os.environ.get("MEDIAPIPE_AUTO_DOWNLOAD_FACE_LANDMARKER", "1").strip().lower()
    if auto_download in {"0", "false", "no", "off"}:
        return None

    target_dir = Path(os.environ.get("MEDIAPIPE_MODEL_CACHE_DIR", "output/models")).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "face_landmarker.task"
    if target_path.is_file() and target_path.stat().st_size > 0:
        return target_path

    url = os.environ.get(
        "MEDIAPIPE_FACE_LANDMARKER_URL",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    )
    tmp_path = target_path.with_suffix(".task.tmp")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            if response.status and response.status >= 400:
                raise RuntimeError(f"HTTP {response.status} while downloading model from: {url}")
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        if not tmp_path.is_file() or tmp_path.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded model file is empty: {tmp_path}")
        tmp_path.replace(target_path)
        return target_path
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


class _MediaPipeTasksFaceMeshAdapter:
    def __init__(self, mp: Any, landmarker: Any) -> None:
        self._mp = mp
        self._landmarker = landmarker

    def process(self, rgb_image: Any) -> Any:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_image)
        results = self._landmarker.detect(mp_image)
        return types.SimpleNamespace(
            multi_face_landmarks=[
                types.SimpleNamespace(landmark=face_landmarks)
                for face_landmarks in (results.face_landmarks or [])
            ]
        )

    def close(self) -> None:
        close = getattr(self._landmarker, "close", None)
        if callable(close):
            close()


def is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "cublas_status_alloc_failed" in message


def clear_cuda_memory() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _resolve_cuda_device_index(device: Optional[Any]) -> int:
    import torch

    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    index = getattr(device, "index", None)
    return torch.cuda.current_device() if index is None else int(index)
