from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import dataclasses
import gc
import hashlib
import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from hairclip_assets import AssetError, ensure_runtime_assets
from runtime_spec import (
    CLIP_INPUT_SIZE,
    EAR_CLASS_INDEX,
    EMBEDDING_REFERENCE_SIZE,
    GENERATOR_MLP_LAYERS,
    GENERATOR_OUTPUT_SIZE,
    HAIR_CLASS_INDEX,
    LATENT_LAYER_COUNT,
    LATENT_STYLE_DIM,
    SEGMENTATION_BACKBONE_SIZE,
    SEGMENTATION_CLASSES,
    scaled_mask_morph_kernel,
    validate_runtime_spec,
)
from utils.face_pipeline import (
    AlignmentResult,
    AlignmentTransform,
    FaceAnalyzer,
    FaceObservation,
    MediaPipeFFHQAligner,
)
from utils.env_loader import load_project_dotenv
from utils.runtime_compat import configure_torch_cuda_arch_list, configure_torch_extension_build_env, safe_torch_load
from utils.sam_masking import (
    DEFAULT_SAM_KEYWORDS,
    SAMMaskingPipeline,
    SAM2Refiner,
    SegmentationResult,
    build_sam2_prompt_hint,
)
from utils.sam2_runtime import create_sam2_predictor_factory, describe_missing_sam2_support


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TREND_DATA_PATH = PROJECT_ROOT / "data" / "trend_hairstyles.json"


LOGGER = logging.getLogger("mirrai.pipeline_optimized")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

validate_runtime_spec()
load_project_dotenv()


def log(message: str) -> None:
    LOGGER.info(message)


def sync_cuda(device: Any) -> None:
    import torch

    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: Any, enabled: bool):
    import torch

    if enabled and getattr(device, "type", None) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def encode_image_base64(image, quality: int = 85) -> str:
    """이미지를 JPEG base64로 인코딩. PNG 대비 10~20배 작아 RunPod 페이로드 한도 초과 방지."""
    buffer = io.BytesIO()
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@dataclasses.dataclass(frozen=True)
class ServicePreset:
    name: str
    w_steps: int
    fs_steps: int
    text_steps: int
    autocast: bool
    use_latent_cache: bool
    target_batch_latency_s: float


PRESETS: Dict[str, ServicePreset] = {
    # realtime: 빠른 응답 우선. text_steps 50으로 상향 (기존 30 → 스타일 변화 부족)
    "realtime": ServicePreset(
        name="realtime",
        w_steps=36,
        fs_steps=30,
        text_steps=60,
        autocast=True,
        use_latent_cache=True,
        target_batch_latency_s=4.0,
    ),
    # balanced: 품질/속도 균형
    "balanced": ServicePreset(
        name="balanced",
        w_steps=50,
        fs_steps=40,
        text_steps=150,
        autocast=True,
        use_latent_cache=True,
        target_batch_latency_s=6.0,
    ),
    # quality: 최고 품질. 스타일 수렴 보장
    "quality": ServicePreset(
        name="quality",
        w_steps=60,
        fs_steps=50,
        text_steps=200,
        autocast=True,
        use_latent_cache=True,
        target_batch_latency_s=9.0,
    ),
}


@dataclasses.dataclass
class ServiceConfig:
    device: str = "cuda"
    preset: str = "realtime"
    precision: str = "fp16"
    enable_tensorrt: bool = False
    enable_torch_compile: bool = False
    async_load: bool = True
    trend_remote_url: Optional[str] = None
    trend_refresh_seconds: int = 1800
    trend_limit: int = 5
    locale: str = "global"
    cache_dir: Path = PROJECT_ROOT / "output" / "service_cache"
    latent_cache_dir: Path = PROJECT_ROOT / "output" / "service_cache" / "latents"
    aligned_cache_dir: Path = PROJECT_ROOT / "output" / "service_cache" / "aligned"
    eta_history_path: Path = PROJECT_ROOT / "output" / "service_cache" / "eta_history.json"
    result_dir: Path = PROJECT_ROOT / "output" / "optimized_results"
    target_output_size: int = GENERATOR_OUTPUT_SIZE
    embedding_reference_size: int = EMBEDDING_REFERENCE_SIZE
    segmentation_backbone_size: int = SEGMENTATION_BACKBONE_SIZE
    clip_input_size: int = CLIP_INPUT_SIZE
    segmentation_classes: int = SEGMENTATION_CLASSES
    hair_class_index: int = HAIR_CLASS_INDEX
    generator_style_dim: int = LATENT_STYLE_DIM
    generator_mlp_layers: int = GENERATOR_MLP_LAYERS
    generator_latent_count: int = LATENT_LAYER_COUNT
    lr_embedding: float = 0.01
    lr_text: float = 0.02               # 0.015 → 0.02: 더 빠른 수렴
    clip_lambda_text: float = 2.0       # 1.5 → 2.0: 텍스트 가이드 강화
    hair_mask_lambda_text: float = 0.65
    latent_anchor_lambda_text: float = 0.003   # ghost 방지: 너무 낮으면 StyleGAN 얼굴 왜곡 → ghost 심화
    no_aug_clip_loss_text: bool = True
    # 합성 품질 옵션
    use_poisson_blend: bool = False     # Poisson은 정렬 오차 있을 때 ghost 유발 → 기본 off
    use_color_match: bool = True        # 생성 hair 색상을 원본 조명에 맞춤
    text_strength: float = 1.0         # clip_lambda 배율 (API 파라미터로 노출)
    sam_keywords: Tuple[str, ...] = dataclasses.field(default_factory=lambda: DEFAULT_SAM_KEYWORDS)
    alignment_intermediate_size: int = GENERATOR_OUTPUT_SIZE * 2
    alignment_upscale_min_size: int = GENERATOR_OUTPUT_SIZE
    alignment_max_upscale_factor: float = 2.0
    max_recommendations: int = 5
    max_parallel_workers: int = 5
    max_gpu_parallel_tasks: int = 3
    estimated_task_vram_gb: float = 3.2
    vram_reserve_gb: float = 7.0
    empty_cache_per_task: bool = True
    enable_sam2: bool = False
    sam2_checkpoint_path: Optional[str] = None
    sam2_auto_download: bool = True
    trend_retriever: Optional["TrendRetriever"] = None
    sam_predictor_factory: Optional[Callable[[], Any]] = None

    def __post_init__(self) -> None:
        if self.target_output_size % self.embedding_reference_size != 0:
            raise ValueError("target_output_size must be divisible by embedding_reference_size.")
        if self.target_output_size % self.segmentation_backbone_size != 0:
            raise ValueError("target_output_size must be divisible by segmentation_backbone_size.")
        if self.segmentation_classes <= self.hair_class_index:
            raise ValueError("hair_class_index must be within segmentation_classes.")
        if self.alignment_intermediate_size < self.target_output_size:
            raise ValueError("alignment_intermediate_size must be >= target_output_size.")
        if self.alignment_upscale_min_size <= 0:
            raise ValueError("alignment_upscale_min_size must be positive.")
        if self.alignment_max_upscale_factor < 1.0:
            raise ValueError("alignment_max_upscale_factor must be >= 1.0.")
        if self.generator_style_dim <= 0 or self.generator_mlp_layers <= 0 or self.generator_latent_count <= 0:
            raise ValueError("Generator dimensions must be positive.")
        if self.enable_sam2 and self.sam_predictor_factory is None:
            self.sam_predictor_factory = create_sam2_predictor_factory(
                device=self.device,
                checkpoint_path=self.sam2_checkpoint_path,
                auto_download=self.sam2_auto_download,
            )
            warning = describe_missing_sam2_support(
                checkpoint_path=self.sam2_checkpoint_path,
                auto_download=self.sam2_auto_download,
            )
            if warning:
                log(f"[SAM2] {warning}")

    def resolved_device(self):
        import torch

        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def preset_config(self) -> ServicePreset:
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset: {self.preset}")
        return PRESETS[self.preset]

    @property
    def use_autocast(self) -> bool:
        return (
            self.precision.lower() in {"fp16", "half"}
            and self.resolved_device().type == "cuda"
            and self.preset_config.autocast
        )


@dataclasses.dataclass(frozen=True)
class TrendQuery:
    face_shape: str
    expression: str
    prompt: Optional[str]
    locale: str
    limit: int = 5


@dataclasses.dataclass(frozen=True)
class TrendDocument:
    id: str
    style_name: str
    description: str
    face_shapes: List[str]
    keywords: List[str]
    regions: List[str]
    occasions: List[str]
    maintenance: str
    popularity_score: float
    freshness_score: float
    source: str
    last_updated: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrendDocument":
        return cls(
            id=str(payload["id"]),
            style_name=str(payload["style_name"]),
            description=str(payload["description"]),
            face_shapes=[str(value).lower() for value in payload.get("face_shapes", [])],
            keywords=[str(value).lower() for value in payload.get("keywords", [])],
            regions=[str(value).lower() for value in payload.get("regions", ["global"])],
            occasions=[str(value).lower() for value in payload.get("occasions", [])],
            maintenance=str(payload.get("maintenance", "medium")).lower(),
            popularity_score=float(payload.get("popularity_score", 0.5)),
            freshness_score=float(payload.get("freshness_score", 0.5)),
            source=str(payload.get("source", "unknown")),
            last_updated=str(payload.get("last_updated", "")),
        )


class TrendRetriever(Protocol):
    def retrieve(self, query: TrendQuery) -> List[TrendDocument]:
        ...


class JSONTrendRetriever:
    def __init__(
        self,
        local_path: Path = DEFAULT_TREND_DATA_PATH,
        remote_url: Optional[str] = None,
        refresh_seconds: int = 1800,
    ) -> None:
        self.local_path = Path(local_path)
        self.remote_url = remote_url
        self.refresh_seconds = refresh_seconds
        self._remote_cache: List[TrendDocument] = []
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def retrieve(self, query: TrendQuery) -> List[TrendDocument]:
        documents = self._load_remote_documents() or self._load_local_documents()
        query_terms = set(self._tokenize([query.face_shape, query.expression, query.prompt or "", query.locale]))
        locale = query.locale.lower()
        scored: List[Tuple[float, TrendDocument]] = []
        for document in documents:
            tokens = set(self._tokenize([document.style_name, document.description, *document.keywords]))
            overlap = len(tokens & query_terms) / max(len(query_terms), 1)
            face_shape_match = 1.0 if query.face_shape.lower() in document.face_shapes else 0.15
            locale_match = 1.0 if locale in document.regions or "global" in document.regions else 0.2
            score = (
                face_shape_match * 0.42
                + document.popularity_score * 0.23
                + document.freshness_score * 0.25
                + overlap * 0.07
                + locale_match * 0.03
            )
            scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[: max(query.limit, 5)]]

    def _load_local_documents(self) -> List[TrendDocument]:
        if not self.local_path.is_file():
            return []
        payload = json.loads(self.local_path.read_text(encoding="utf-8"))
        return [TrendDocument.from_dict(item) for item in payload]

    def _load_remote_documents(self) -> List[TrendDocument]:
        if not self.remote_url:
            return []
        now = time.time()
        with self._lock:
            if self._remote_cache and now < self._expires_at:
                return list(self._remote_cache)
        try:
            request = urllib.request.Request(
                self.remote_url,
                headers={"User-Agent": "MirrAI-Pipeline-Optimized/1.0"},
            )
            with urllib.request.urlopen(request, timeout=2.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            documents = [TrendDocument.from_dict(item) for item in payload]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError):
            return []
        with self._lock:
            self._remote_cache = documents
            self._expires_at = now + self.refresh_seconds
        return list(documents)

    @staticmethod
    def _tokenize(parts: Iterable[str]) -> List[str]:
        tokens: List[str] = []
        for part in parts:
            lowered = part.lower().replace("/", " ").replace("-", " ")
            tokens.extend(token for token in lowered.split() if token)
        return tokens


class CallableTrendRetriever:
    def __init__(self, handler: Callable[[TrendQuery], Sequence[Mapping[str, Any]]]) -> None:
        self.handler = handler

    def retrieve(self, query: TrendQuery) -> List[TrendDocument]:
        return [TrendDocument.from_dict(item) for item in self.handler(query)]


class Neo4jTrendRetriever(CallableTrendRetriever):
    pass


class VectorDBTrendRetriever:
    def __init__(
        self,
        similarity_search: Callable[[str, int, Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    ) -> None:
        self.similarity_search = similarity_search

    def retrieve(self, query: TrendQuery) -> List[TrendDocument]:
        query_text = " ".join(
            value
            for value in [query.face_shape, query.expression, query.prompt or "", query.locale]
            if value
        )
        filters = {"face_shape": query.face_shape, "locale": query.locale}
        return [
            TrendDocument.from_dict(item)
            for item in self.similarity_search(query_text, max(query.limit, 5), filters)
        ]


class TrendRecommendationEngine:
    FACE_BALANCE_GUIDANCE = {
        "round": "elongated silhouette, crown volume, cheek-slimming face framing",
        "oval": "balanced silhouette, soft texture, natural proportion",
        "square": "jaw-softening contour, airy texture, curved outline",
        "heart": "forehead-softening fringe, jaw-balancing fullness, soft perimeter",
        "oblong": "width-building volume, reduced vertical emphasis, lateral movement",
    }

    def __init__(self, retriever: TrendRetriever) -> None:
        self.retriever = retriever

    def recommend(
        self,
        face_shape: str,
        expression: str = "neutral",
        prompt: Optional[str] = None,
        locale: str = "global",
        limit: int = 5,
    ) -> Dict[str, Any]:
        query = TrendQuery(
            face_shape=face_shape,
            expression=expression,
            prompt=prompt,
            locale=locale,
            limit=max(limit, 5),
        )
        documents = self.retriever.retrieve(query)
        styles = [self._document_to_payload(document, query) for document in documents[: max(limit, 5)]]
        return {
            "face_shape": face_shape,
            "expression": expression,
            "query": dataclasses.asdict(query),
            "styles": styles,
            "retriever": self.retriever.__class__.__name__,
        }

    def _document_to_payload(self, document: TrendDocument, query: TrendQuery) -> Dict[str, Any]:
        score = round(document.popularity_score * 0.55 + document.freshness_score * 0.45, 4)
        return {
            "id": document.id,
            "style_name": document.style_name,
            "description": document.description,
            "fit_reason": self._build_fit_reason(query.face_shape, query.expression, document),
            "maintenance": document.maintenance,
            "score": score,
            "source": document.source,
            "last_updated": document.last_updated,
            "keywords": document.keywords,
            "hairclip_prompt": self._build_hairclip_prompt(document, query.face_shape, query.prompt),
        }

    def _build_fit_reason(self, face_shape: str, expression: str, document: TrendDocument) -> str:
        reasons: List[str] = []
        if face_shape.lower() in document.face_shapes:
            reasons.append(f"{face_shape} face balance support")
        if expression.lower() == "smile":
            reasons.append("keeps soft framing for an open expression")
        if "volume" in document.keywords or "layered" in document.keywords:
            reasons.append("supports controllable silhouette volume")
        if not reasons:
            reasons.append("ranked highly in the active trend corpus")
        return ", ".join(reasons)

    def _build_hairclip_prompt(
        self,
        document: TrendDocument,
        face_shape: str,
        user_prompt: Optional[str],
    ) -> str:
        guidance = self.FACE_BALANCE_GUIDANCE.get(face_shape.lower(), "balanced silhouette")
        parts = [
            document.style_name.lower(),
            document.description.lower(),
            guidance,
            "realistic salon-quality hairstyle edit",
            "preserve identity",
        ]
        if user_prompt:
            parts.append(user_prompt.strip().lower())
        return ", ".join(dict.fromkeys(part for part in parts if part))


class TrendKnowledgeBase:
    def __init__(
        self,
        local_path: Path = DEFAULT_TREND_DATA_PATH,
        remote_url: Optional[str] = None,
        refresh_seconds: int = 1800,
        retriever: Optional[TrendRetriever] = None,
    ) -> None:
        self.retriever = retriever or JSONTrendRetriever(
            local_path=local_path,
            remote_url=remote_url,
            refresh_seconds=refresh_seconds,
        )
        self.engine = TrendRecommendationEngine(self.retriever)

    def recommend(
        self,
        face_shape: str,
        expression: str,
        prompt: Optional[str] = None,
        locale: str = "global",
        limit: int = 5,
    ) -> Dict[str, Any]:
        return self.engine.recommend(
            face_shape=face_shape,
            expression=expression,
            prompt=prompt,
            locale=locale,
            limit=limit,
        )

    def search(
        self,
        face_shape: str,
        expression: str,
        prompt: Optional[str] = None,
        locale: str = "global",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        return self.recommend(
            face_shape=face_shape,
            expression=expression,
            prompt=prompt,
            locale=locale,
            limit=limit,
        )["styles"]


def get_recommendations(
    face_shape: str,
    trend_kb: Optional[TrendKnowledgeBase] = None,
    expression: str = "neutral",
    prompt: Optional[str] = None,
    locale: str = "global",
    limit: int = 5,
) -> Dict[str, Any]:
    knowledge_base = trend_kb or TrendKnowledgeBase()
    return knowledge_base.recommend(
        face_shape=face_shape,
        expression=expression,
        prompt=prompt,
        locale=locale,
        limit=limit,
    )


class StageProfiler:
    def __init__(self, device: Any) -> None:
        self.device = device
        self.timings_ms: Dict[str, float] = {}
        self.vram_mb: Dict[str, Dict[str, float]] = {}

    @contextlib.contextmanager
    def measure(self, stage_name: str):
        import torch

        sync_cuda(self.device)
        started_at = time.perf_counter()
        if self.device.type == "cuda":
            before_allocated = torch.cuda.memory_allocated(self.device)
            before_reserved = torch.cuda.memory_reserved(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            before_allocated = 0
            before_reserved = 0
        try:
            yield
        finally:
            sync_cuda(self.device)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self.timings_ms[stage_name] = round(elapsed_ms, 3)
            if self.device.type == "cuda":
                current_allocated = torch.cuda.memory_allocated(self.device)
                current_reserved = torch.cuda.memory_reserved(self.device)
                peak_allocated = torch.cuda.max_memory_allocated(self.device)
                self.vram_mb[stage_name] = {
                    "allocated_mb": round(current_allocated / (1024 ** 2), 2),
                    "reserved_mb": round(current_reserved / (1024 ** 2), 2),
                    "delta_allocated_mb": round((current_allocated - before_allocated) / (1024 ** 2), 2),
                    "delta_reserved_mb": round((current_reserved - before_reserved) / (1024 ** 2), 2),
                    "peak_allocated_mb": round(peak_allocated / (1024 ** 2), 2),
                }

    def snapshot(self) -> Dict[str, float]:
        import torch

        if self.device.type != "cuda":
            return {}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated(self.device) / (1024 ** 2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(self.device) / (1024 ** 2), 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2), 2),
        }


class EtaEstimator:
    def __init__(self, history_path: Path) -> None:
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def estimate(
        self,
        preset: ServicePreset,
        runtime_hot: bool,
        cache_hit: bool,
        batch_size: int,
        parallel_slots: int,
    ) -> Dict[str, Any]:
        history = self._load()
        base_embedding_ms = 35.0 if cache_hit else max(220.0, (preset.w_steps * 8.0 + preset.fs_steps * 8.5))
        per_style_ms = max(260.0, preset.text_steps * 15.0)
        waves = max(1, (batch_size + max(parallel_slots, 1) - 1) // max(parallel_slots, 1))
        total_ms = (
            history.get("analysis", 35.0)
            + history.get("alignment", 40.0)
            + history.get("recommendation", 8.0)
            + (0.0 if runtime_hot else history.get("runtime_wait", 350.0))
            + base_embedding_ms
            + per_style_ms * waves
        )
        return {
            "estimated_total_ms": round(total_ms, 3),
            "assumptions": {
                "runtime_hot": runtime_hot,
                "cache_hit": cache_hit,
                "batch_size": batch_size,
                "parallel_slots": parallel_slots,
                "preset": preset.name,
            },
        }

    def record(self, stage_timings: Mapping[str, float]) -> None:
        with self._lock:
            history = self._load()
            alpha = 0.35
            for stage_name, elapsed_ms in stage_timings.items():
                previous = float(history.get(stage_name, elapsed_ms))
                history[stage_name] = round(previous * (1 - alpha) + float(elapsed_ms) * alpha, 3)
            self.history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> Dict[str, Any]:
        if not self.history_path.is_file():
            return {}
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}


class VRAMTaskController:
    def __init__(self, device: Any, config: ServiceConfig) -> None:
        self.device = device
        self.config = config
        self.parallel_slots = self._resolve_parallel_slots()
        self._semaphore = threading.BoundedSemaphore(self.parallel_slots)

    @contextlib.contextmanager
    def acquire(self):
        self._semaphore.acquire()
        try:
            yield self.parallel_slots
        finally:
            self._semaphore.release()

    def _resolve_parallel_slots(self) -> int:
        import torch

        hard_cap = max(1, min(self.config.max_parallel_workers, self.config.max_gpu_parallel_tasks))
        if self.device.type != "cuda":
            return 1
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            usable_gb = max(0.0, min(free_gb, total_gb) - self.config.vram_reserve_gb)
            slots = int(usable_gb // self.config.estimated_task_vram_gb)
            return max(1, min(hard_cap, slots if slots > 0 else 1))
        except Exception:
            return max(1, min(hard_cap, 2))


class LatentCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, aligned_image_path: str, preset_name: str) -> str:
        file_bytes = Path(aligned_image_path).read_bytes()
        return hashlib.sha256(file_bytes + preset_name.encode("utf-8")).hexdigest()

    def get(self, aligned_image_path: str, preset_name: str) -> Optional[Tuple[Any, Any]]:
        import torch

        cache_path = self.cache_dir / f"{self._key(aligned_image_path, preset_name)}.pt"
        if not cache_path.is_file():
            return None
        payload = safe_torch_load(cache_path, map_location="cpu")
        return payload["src_latent"], payload["latent_F"]

    def set(self, aligned_image_path: str, preset_name: str, src_latent: Any, latent_f: Any) -> None:
        import torch

        cache_path = self.cache_dir / f"{self._key(aligned_image_path, preset_name)}.pt"
        payload = {
            "src_latent": src_latent.detach().cpu(),
            "latent_F": latent_f.detach().cpu(),
            "created_at": time.time(),
        }
        torch.save(payload, cache_path)

    def contains(self, aligned_image_path: str, preset_name: str) -> bool:
        cache_path = self.cache_dir / f"{self._key(aligned_image_path, preset_name)}.pt"
        return cache_path.is_file()


class FastEmbeddingOptimizer:
    def __init__(self, opts: Any, generator: Any, mean_latent: Any, device: Any, use_autocast: bool) -> None:
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms

        from criteria.embedding_loss import EmbeddingLossBuilder
        from utils.bicubic import BicubicDownSample

        self.opts = opts
        self.generator = generator
        self.generator_dtype = next(generator.parameters()).dtype
        self.mean_latent = mean_latent.detach().float()
        self.device = device
        self.use_autocast = use_autocast
        self.image_loader = Image
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.loss_builder = EmbeddingLossBuilder(opts).eval()
        self.downsample = BicubicDownSample(factor=opts.output_size // opts.embedding_size)
        pca_model = np.load(opts.ffhq_pca_path)
        self.x_mean = torch.from_numpy(pca_model["X_mean"]).float().to(device)
        self.x_comp = torch.from_numpy(pca_model["X_comp"]).float().to(device)
        self.x_stdev = torch.from_numpy(pca_model["X_stdev"]).float().to(device)

    def invert_image_in_fs(self, image_path: str) -> Tuple[Any, Any]:
        import torch

        ref_high, ref_low = self._prepare_reference(image_path)
        latent_w = self._invert_in_w(ref_high, ref_low)
        with torch.no_grad():
            with autocast_context(self.device, self.use_autocast):
                latent_f, _ = self.generator(
                    [latent_w.to(self.generator_dtype)],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=0,
                    end_layer=3,
                )
        latent_s = torch.nn.Parameter(latent_w.detach().clone().float())
        latent_f_param = torch.nn.Parameter(latent_f.detach().clone().float())
        grad_mask = torch.ones_like(latent_s)
        grad_mask[:, :7, :] = 0
        latent_s.register_hook(lambda grad: grad * grad_mask)
        optimizer = torch.optim.Adam([latent_s, latent_f_param], lr=self.opts.lr_embedding)

        for _ in range(self.opts.FS_steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.use_autocast):
                gen_im, _ = self.generator(
                    [latent_s.to(self.generator_dtype)],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=4,
                    end_layer=8,
                    layer_in=latent_f_param.to(self.generator_dtype),
                )
            loss = self._embedding_loss(
                ref_high=ref_high,
                ref_low=ref_low,
                generated=gen_im.float(),
                latent=latent_s.float(),
            )
            loss.backward()
            optimizer.step()
        return latent_s.detach(), latent_f_param.detach()

    def _invert_in_w(self, ref_high: Any, ref_low: Any):
        import torch

        base_latent = self.mean_latent[0:1].detach().clone().float().to(self.device)
        latent_w = torch.nn.Parameter(base_latent)
        optimizer = torch.optim.Adam([latent_w], lr=self.opts.lr_embedding)

        for _ in range(self.opts.W_steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.use_autocast):
                gen_im, _ = self.generator(
                    [latent_w.to(self.generator_dtype)],
                    input_is_latent=True,
                    return_latents=False,
                    randomize_noise=False,
                )
            loss = self._embedding_loss(
                ref_high=ref_high,
                ref_low=ref_low,
                generated=gen_im.float(),
                latent=latent_w.float(),
            )
            loss.backward()
            optimizer.step()
        return latent_w.detach()

    def _prepare_reference(self, image_path: str):
        ref_image = self.image_loader.open(image_path).convert("RGB")
        ref_low = self.image_transform(
            ref_image.resize((self.opts.embedding_size, self.opts.embedding_size), self.image_loader.LANCZOS)
        ).unsqueeze(0)
        ref_high = self.image_transform(
            ref_image.resize((self.opts.output_size, self.opts.output_size), self.image_loader.LANCZOS)
        ).unsqueeze(0)
        return ref_high.to(self.device), ref_low.to(self.device)

    def _embedding_loss(self, ref_high: Any, ref_low: Any, generated: Any, latent: Any):
        import torch

        generated_low = self.downsample(generated.float())
        loss, _ = self.loss_builder(
            ref_im_H=ref_high.float(),
            ref_im_L=ref_low.float(),
            gen_im_H=generated.float(),
            gen_im_L=generated_low.float(),
        )
        latent_p_norm = (
            (torch.nn.LeakyReLU(negative_slope=5)(latent) - self.x_mean).bmm(self.x_comp.T.unsqueeze(0))
            / self.x_stdev
        )
        return loss + self.opts.p_norm_lambda_embedding * latent_p_norm.pow(2).mean()


class FastTextStyleEditor:
    def __init__(self, opts: Any, generator: Any, seg: Any, device: Any, use_autocast: bool) -> None:
        import torch

        from criteria.clip_loss import AugCLIPLoss, CLIPLoss

        self.opts = opts
        self.generator = generator
        self.seg = seg
        self.device = device
        self.use_autocast = use_autocast
        self.generator_dtype = next(generator.parameters()).dtype
        self.clip_loss = (
            CLIPLoss(device=device, clip_input_size=opts.clip_input_size)
            if opts.no_aug_clip_loss_text
            else AugCLIPLoss(device=device, clip_input_size=opts.clip_input_size)
        )
        for parameter in self.clip_loss.parameters():
            parameter.requires_grad = False
        self.mask_loss = torch.nn.BCEWithLogitsLoss()

    def optimize(self, source_latent: Any, text_prompt: str, target_hair_mask: Optional[Any] = None):
        import torch
        import torch.nn.functional as F

        source_latent = source_latent.detach().float()
        latent = torch.nn.Parameter(source_latent.clone())
        anchor = source_latent.clone()
        grad_mask = torch.zeros_like(latent)
        grad_mask[:, :7, :] = 1
        latent.register_hook(lambda grad: grad * grad_mask)
        optimizer = torch.optim.Adam([latent], lr=self.opts.lr_text)

        for _ in range(self.opts.steps_text):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.use_autocast):
                generated, _ = self.generator(
                    [latent.to(self.generator_dtype)],
                    input_is_latent=True,
                    randomize_noise=False,
                )
            clip_loss = self.clip_loss(generated.float(), text_prompt)
            loss = self.opts.clip_lambda_text * clip_loss
            if target_hair_mask is not None:
                with autocast_context(self.device, self.use_autocast):
                    _, logits = self.seg(generated)
                hair_logits = logits[:, self.opts.hair_class_index:self.opts.hair_class_index + 1].float()
                resized_target = F.interpolate(
                    target_hair_mask.float(),
                    size=hair_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = loss + self.opts.hair_mask_lambda_text * self.mask_loss(hair_logits, resized_target)
            loss = loss + self.opts.latent_anchor_lambda_text * (latent[:, 7:, :] - anchor[:, 7:, :]).pow(2).mean()
            loss.backward()
            optimizer.step()
        return latent.detach()


@dataclasses.dataclass
class RuntimeBundle:
    generator: Any
    seg: Any
    bald_mapper: Any
    bald_alpha: float
    embedding: FastEmbeddingOptimizer
    text_editor: FastTextStyleEditor
    mean_latent: Any
    generator_dtype: Any


class ProductionRuntime:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.device = config.resolved_device()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._future: Optional[concurrent.futures.Future[RuntimeBundle]] = None
        self._runtime: Optional[RuntimeBundle] = None
        self._lock = threading.Lock()

    def warmup_async(self):
        with self._lock:
            if self._future is None:
                self._future = self._executor.submit(self._load_sync)
        return self._future

    def get(self) -> RuntimeBundle:
        if self._runtime is not None:
            return self._runtime
        if self._future is None:
            self._future = self._executor.submit(self._load_sync)
        self._runtime = self._future.result()
        return self._runtime

    def is_hot(self) -> bool:
        return self._runtime is not None

    def _load_sync(self) -> RuntimeBundle:
        import sys
        import torch

        if self.device.type != "cuda":
            raise RuntimeError(
                "The optimized HairCLIPv2 runtime requires CUDA. "
                "Recommendation-only mode may run on CPU."
            )

        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        configure_torch_cuda_arch_list(self.device)
        configure_torch_extension_build_env(PROJECT_ROOT)

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from models.bald_proxy.networks.level_mapper import LevelMapper
        from models.face_parsing.model import BiSeNet
        from models.stylegan2.model import Generator
        from utils.options import Options

        try:
            asset_layout = ensure_runtime_assets(restore_from_legacy=True)
        except AssetError as exc:
            raise RuntimeError(
                "HairCLIPv2 runtime assets are incomplete. "
                "Restore the local RunPod backup or place the required files under pretrained_models/."
            ) from exc

        opts = Options().parse(jupyter=True)
        opts.stylegan_path = str(asset_layout.files["ffhq.pt"])
        opts.seg_path = str(asset_layout.files["seg.pth"])
        opts.bald_path = str(asset_layout.files["bald_proxy.pt"])
        opts.ffhq_pca_path = str(asset_layout.files["ffhq_PCA.npz"])
        opts.W_steps = self.config.preset_config.w_steps
        opts.FS_steps = self.config.preset_config.fs_steps
        opts.steps_text = self.config.preset_config.text_steps
        opts.visual_num_text = 1
        opts.output_size = self.config.target_output_size
        opts.embedding_size = self.config.embedding_reference_size
        opts.segmentation_size = self.config.segmentation_backbone_size
        opts.clip_input_size = self.config.clip_input_size
        opts.segmentation_classes = self.config.segmentation_classes
        opts.hair_class_index = self.config.hair_class_index
        opts.generator_style_dim = self.config.generator_style_dim
        opts.generator_mlp_layers = self.config.generator_mlp_layers
        opts.generator_latent_count = self.config.generator_latent_count
        opts.device = str(self.device)
        opts.lr_embedding = self.config.lr_embedding
        opts.lr_text = self.config.lr_text
        opts.clip_lambda_text = self.config.clip_lambda_text * self.config.text_strength
        opts.hair_mask_lambda_text = self.config.hair_mask_lambda_text
        opts.no_aug_clip_loss_text = self.config.no_aug_clip_loss_text
        opts.latent_anchor_lambda_text = self.config.latent_anchor_lambda_text

        ckpt = safe_torch_load(opts.stylegan_path, map_location="cpu")
        generator = Generator(opts.output_size, opts.generator_style_dim, opts.generator_mlp_layers)
        generator.load_state_dict(ckpt["g_ema"], strict=False)
        generator.eval().requires_grad_(False).to(self.device)

        mean_latent = (
            ckpt["latent_avg"]
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, opts.generator_latent_count, 1)
            .clone()
            .detach()
            .to(self.device)
        )

        seg = BiSeNet(
            n_classes=opts.segmentation_classes,
            output_size=opts.output_size,
            input_size=opts.segmentation_size,
        )
        seg.load_state_dict(safe_torch_load(opts.seg_path, map_location="cpu"), strict=False)
        seg.eval().requires_grad_(False).to(self.device)

        if self.config.enable_tensorrt:
            seg = self._maybe_compile_segmentation_trt(seg)

        bald_ckpt = safe_torch_load(opts.bald_path, map_location="cpu")
        bald_mapper = LevelMapper(input_dim=opts.generator_style_dim)
        bald_mapper.load_state_dict(bald_ckpt["state_dict"], strict=True)
        bald_mapper.eval().requires_grad_(False).to(self.device)
        bald_alpha = float(bald_ckpt["alpha"]) * 1.2

        if self.config.use_autocast:
            generator.half()
            bald_mapper.half()
            mean_latent = mean_latent.half()
            if not self.config.enable_tensorrt:
                seg.half()

        if self.config.enable_torch_compile and hasattr(torch, "compile"):
            try:
                generator = torch.compile(generator, mode="reduce-overhead")
            except Exception as exc:
                log(f"[Runtime] torch.compile(generator) was skipped: {exc}")

        embedding = FastEmbeddingOptimizer(
            opts=opts,
            generator=generator,
            mean_latent=mean_latent,
            device=self.device,
            use_autocast=self.config.use_autocast,
        )
        text_editor = FastTextStyleEditor(
            opts=opts,
            generator=generator,
            seg=seg,
            device=self.device,
            use_autocast=self.config.use_autocast,
        )

        return RuntimeBundle(
            generator=generator,
            seg=seg,
            bald_mapper=bald_mapper,
            bald_alpha=bald_alpha,
            embedding=embedding,
            text_editor=text_editor,
            mean_latent=mean_latent,
            generator_dtype=next(generator.parameters()).dtype,
        )

    def _maybe_compile_segmentation_trt(self, seg: Any) -> Any:
        import torch

        try:
            import torch_tensorrt
        except ImportError:
            log("[Runtime] torch_tensorrt is not installed. PyTorch segmentation will be used.")
            return seg
        try:
            dtype = torch.half if self.config.use_autocast else torch.float
            return torch_tensorrt.compile(
                seg,
                inputs=[torch_tensorrt.Input((1, 3, self.config.target_output_size, self.config.target_output_size), dtype=dtype)],
                enabled_precisions={dtype},
                truncate_long_and_double=True,
            )
        except Exception as exc:
            log(f"[Runtime] TensorRT segmentation compile failed: {exc}")
            return seg


class MirrAIOptimizedPipeline:
    def __init__(self, config: Optional[ServiceConfig] = None) -> None:
        self.config = config or ServiceConfig()
        self.device = self.config.resolved_device()
        self.face_analyzer = FaceAnalyzer()
        self.aligner = MediaPipeFFHQAligner(
            output_size=self.config.target_output_size,
            intermediate_size=self.config.alignment_intermediate_size,
            min_source_size=self.config.alignment_upscale_min_size,
            max_upscale_factor=self.config.alignment_max_upscale_factor,
        )
        self.trend_kb = TrendKnowledgeBase(
            local_path=DEFAULT_TREND_DATA_PATH,
            remote_url=self.config.trend_remote_url,
            refresh_seconds=self.config.trend_refresh_seconds,
            retriever=self.config.trend_retriever,
        )
        self.runtime = ProductionRuntime(self.config)
        self.latent_cache = LatentCache(self.config.latent_cache_dir)
        self.eta_estimator = EtaEstimator(self.config.eta_history_path)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.latent_cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.aligned_cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.result_dir.mkdir(parents=True, exist_ok=True)
        if self.config.async_load:
            self.runtime.warmup_async()

    def close(self) -> None:
        self.face_analyzer.close()

    def warmup_async(self) -> None:
        self.runtime.warmup_async()

    def analyze_face(self, image_input: Any) -> Optional[Dict[str, Any]]:
        image_bgr = self._load_image_bgr(image_input)
        observation = self.face_analyzer.inspect(image_bgr)
        return observation.to_payload() if observation else None

    def align_face(self, image_input: Any) -> List[AlignmentResult]:
        import cv2
        image_bgr = self._load_image_bgr(image_input)
        observation = self.face_analyzer.inspect(image_bgr)
        if not observation:
            return []
        
        transform = self.aligner.build_transform(image_bgr, observation)
        alignment_result = self.aligner.align(image_bgr, observation, transform=transform)
        return [alignment_result]

    def recommend_styles(
        self,
        face_analysis: Optional[Dict[str, Any]],
        prompt: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not face_analysis:
            return None
        return self.trend_kb.recommend(
            face_shape=face_analysis["face_shape"],
            expression=face_analysis["expression"],
            prompt=prompt,
            locale=self.config.locale,
            limit=limit or self.config.trend_limit,
        )

    async def generate_multiple_styles_async(self, *args, **kwargs) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.generate_multiple_styles(*args, **kwargs))

    def estimate_eta(
        self,
        image_path: str,
        text_prompt: str = "",
        skip_alignment: bool = False,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        image_path = str(Path(image_path).resolve())
        image_bgr = self._load_image_bgr(image_path)
        observation = self.face_analyzer.inspect(image_bgr)
        if observation is None:
            raise ValueError(f"Unable to detect a face in: {image_path}")
        aligned_path = image_path if skip_alignment else self._aligned_cache_path(image_path)
        cache_hit = Path(aligned_path).is_file() and self.latent_cache.contains(aligned_path, self.config.preset)
        task_controller = VRAMTaskController(self.device, self.config)
        return self.eta_estimator.estimate(
            preset=self.config.preset_config,
            runtime_hot=self.runtime.is_hot(),
            cache_hit=cache_hit,
            batch_size=limit or self.config.max_recommendations,
            parallel_slots=task_controller.parallel_slots,
        )

    def prepare_identity(self, image_path: str, skip_alignment: bool = False) -> Dict[str, Any]:
        image_path = str(Path(image_path).resolve())
        image_bgr = self._load_image_bgr(image_path)
        observation = self.face_analyzer.inspect(image_bgr)
        if observation is None:
            raise ValueError(f"Unable to detect a face in: {image_path}")
        if skip_alignment:
            aligned_path = image_path
        else:
            aligned_path, _, _ = self._align_and_cache(image_path, image_bgr, observation)
        runtime = self.runtime.get()
        cache_hit_before = self.latent_cache.contains(aligned_path, self.config.preset)
        self._get_cached_or_invert(runtime, aligned_path)
        return {
            "image_path": image_path,
            "aligned_path": aligned_path,
            "cache_hit_before": cache_hit_before,
            "cache_hit_after": True,
            "face_analysis": observation.to_payload(),
        }

    def run(
        self,
        image_path: str,
        text_prompt: str,
        skip_alignment: bool = False,
        return_base64: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_multiple_styles(
            image_path=image_path,
            text_prompt=text_prompt,
            skip_alignment=skip_alignment,
            return_base64=return_base64,
            output_dir=output_dir,
        )

    def generate_multiple_styles(
        self,
        image_path: str,
        text_prompt: str = "",
        skip_alignment: bool = False,
        return_base64: bool = False,
        output_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        image_path = str(Path(image_path).resolve())
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        profiler = StageProfiler(self.device)
        started_at = time.perf_counter()
        runtime_hot_before = self.runtime.is_hot()

        with profiler.measure("analysis"):
            image_bgr = self._load_image_bgr(image_path)
            observation = self.face_analyzer.inspect(image_bgr)
            if observation is None:
                raise ValueError(f"Unable to detect a face in: {image_path}")
            face_analysis = observation.to_payload()

        with profiler.measure("recommendation"):
            recommendation = self.recommend_styles(
                face_analysis=face_analysis,
                prompt=text_prompt,
                limit=limit or self.config.max_recommendations,
            )
            styles = (recommendation or {}).get("styles", [])[: limit or self.config.max_recommendations]
            if not styles:
                raise RuntimeError("No hairstyle recommendations were produced.")

        if skip_alignment:
            aligned_path = image_path
            aligned_rgb = self._load_image_rgb(image_path)
            alignment_transform = None
        else:
            with profiler.measure("alignment"):
                aligned_path, aligned_rgb, alignment_transform = self._align_and_cache(image_path, image_bgr, observation)

        task_controller = VRAMTaskController(self.device, self.config)
        eta = self.eta_estimator.estimate(
            preset=self.config.preset_config,
            runtime_hot=runtime_hot_before,
            cache_hit=self.latent_cache.contains(aligned_path, self.config.preset),
            batch_size=len(styles),
            parallel_slots=task_controller.parallel_slots,
        )

        with profiler.measure("runtime_wait"):
            runtime = self.runtime.get()

        with profiler.measure("embedding"):
            src_latent, latent_f = self._get_cached_or_invert(runtime, aligned_path)
            source_image_tensor = self._vectorized_image_tensor(aligned_rgb)
            masking_pipeline = SAMMaskingPipeline(
                seg=runtime.seg,
                device=self.device,
                use_autocast=self.config.use_autocast,
                predictor_factory=self.config.sam_predictor_factory,
                hair_class_index=self.config.hair_class_index,
                autocast_factory=autocast_context,
            )
            # SAM3 → MediaPipe landmarks (remapped to aligned space) → HairCLIPv2-ready segmentation
            segmentation = masking_pipeline.segment(
                image_tensor=source_image_tensor,
                image_rgb=aligned_rgb,
                face_observation=observation,
                prompt=text_prompt,
                alignment_transform=alignment_transform,
            )

        request_dir = self._request_output_dir(image_path=image_path, output_dir=output_dir)

        with profiler.measure("parallel_generation"):
            results = self._run_parallel_style_tasks(
                runtime=runtime,
                styles=styles,
                src_latent=src_latent,
                latent_f=latent_f,
                original_rgb=self._bgr_to_rgb(image_bgr),
                aligned_rgb=aligned_rgb,
                alignment_transform=alignment_transform,
                face_observation=observation,
                segmentation=segmentation,
                output_dir=request_dir,
                return_base64=return_base64,
                task_controller=task_controller,
            )

        total_ms = (time.perf_counter() - started_at) * 1000.0
        profiler.timings_ms["total"] = round(total_ms, 3)
        self.eta_estimator.record(profiler.timings_ms)

        self._release_temporaries(
            image_bgr=image_bgr,
            aligned_rgb=aligned_rgb,
            source_image_tensor=source_image_tensor,
            src_latent=src_latent,
            latent_f=latent_f,
            segmentation=segmentation,
        )

        return {
            "face_analysis": face_analysis,
            "recommendation": recommendation,
            "results": results,
            "performance": {
                "stages_ms": profiler.timings_ms,
                "vram_mb": {
                    "stages": profiler.vram_mb,
                    "after_run": profiler.snapshot(),
                },
                "parallel_slots": task_controller.parallel_slots,
                "shared_runtime_weights": True,
                "target_batch_latency_s": self.config.preset_config.target_batch_latency_s,
            },
            "eta": eta,
            "output_dir": str(request_dir),
        }

    def _run_parallel_style_tasks(
        self,
        runtime: RuntimeBundle,
        styles: Sequence[Mapping[str, Any]],
        src_latent: Any,
        latent_f: Any,
        original_rgb: Any,
        aligned_rgb: Any,
        alignment_transform: Optional[AlignmentTransform],
        face_observation: FaceObservation,
        segmentation: SegmentationResult,
        output_dir: Path,
        return_base64: bool,
        task_controller: VRAMTaskController,
    ) -> List[Dict[str, Any]]:
        max_workers = max(1, min(len(styles), self.config.max_parallel_workers))
        futures: List[concurrent.futures.Future] = []
        results: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, style in enumerate(styles):
                futures.append(
                    executor.submit(
                        self._generate_style_task,
                        index,
                        style,
                        runtime,
                        src_latent,
                        latent_f,
                        original_rgb,
                        aligned_rgb,
                        alignment_transform,
                        face_observation,
                        segmentation,
                        output_dir,
                        return_base64,
                        task_controller,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["rank"])
        return results

    def _generate_style_task(
        self,
        rank: int,
        style: Mapping[str, Any],
        runtime: RuntimeBundle,
        src_latent: Any,
        latent_f: Any,
        original_rgb: Any,
        aligned_rgb: Any,
        alignment_transform: Optional[AlignmentTransform],
        face_observation: FaceObservation,
        segmentation: SegmentationResult,
        output_dir: Path,
        return_base64: bool,
        task_controller: VRAMTaskController,
    ) -> Dict[str, Any]:
        import PIL.Image
        import torch
        import numpy as np

        from scripts.feature_blending import hairstyle_feature_blending
        from utils.image_utils import process_display_input

        started_at = time.perf_counter()
        before_snapshot = self._current_vram_snapshot()

        prompt = str(style["hairclip_prompt"])
        with task_controller.acquire():
            text_latent = runtime.text_editor.optimize(
                source_latent=src_latent,
                text_prompt=prompt,
                target_hair_mask=segmentation.hair_mask if segmentation.sam_refined else None,
            )

            with torch.no_grad():
                src_latent_runtime = src_latent.to(self.device).to(runtime.generator_dtype)
                latent_f_runtime = latent_f.to(self.device).to(runtime.generator_dtype)
                text_latent_runtime = text_latent.to(self.device).to(runtime.generator_dtype)
                latent_bald = src_latent_runtime.clone()
                latent_bald[:, :8, :] += runtime.bald_alpha * runtime.bald_mapper(src_latent_runtime)
                _, result_tensor = hairstyle_feature_blending(
                    generator=runtime.generator,
                    seg=runtime.seg,
                    src_latent=src_latent_runtime,
                    src_feature=latent_f_runtime,
                    visual_mask=segmentation.parsing_mask.long(),
                    latent_bald=latent_bald,
                    latent_global=text_latent_runtime,
                    latent_local=None,
                    local_blending_mask=None,
                )
                edited_aligned_rgb = process_display_input(result_tensor.float())
                composited_to_original = alignment_transform is not None
                log(
                    f"[composite] rank={rank}  composited_to_original={composited_to_original}  "
                    f"sam_refined={segmentation.sam_refined}  "
                    f"aligned_shape={edited_aligned_rgb.shape}  "
                    f"original_shape={original_rgb.shape if original_rgb is not None else 'None'}"
                )
                final_rgb = (
                    self._composite_aligned_result_to_original(
                        runtime=runtime,
                        original_rgb=original_rgb,
                        edited_aligned_rgb=edited_aligned_rgb,
                        edited_result_tensor=result_tensor,
                        segmentation=segmentation,
                        alignment_transform=alignment_transform,
                    )
                    if composited_to_original
                    else edited_aligned_rgb
                )
                log(f"[composite] rank={rank}  final_shape={final_rgb.shape}")
                result_image = PIL.Image.fromarray(final_rgb)

            style_slug = self._slugify(f"{rank + 1}_{style['style_name']}")
            output_path = output_dir / f"{style_slug}.png"
            result_image.save(output_path)

            payload = {
                "rank": rank,
                "style_name": style["style_name"],
                "description": style["description"],
                "fit_reason": style["fit_reason"],
                "maintenance": style["maintenance"],
                "source": style["source"],
                "last_updated": style["last_updated"],
                "score": style["score"],
                "prompt": prompt,
                "masking": {
                    "source": segmentation.source,
                    "sam_refined": segmentation.sam_refined,
                    "sam_prompt": segmentation.sam_prompt,
                },
                "output_frame": "original" if composited_to_original else "aligned",
                "composited_to_original": composited_to_original,
                "image_path": str(output_path.resolve()),
                "image_base64": encode_image_base64(result_image) if return_base64 else None,
            }

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        after_snapshot = self._current_vram_snapshot()
        payload["report"] = {
            "latency_ms": round(elapsed_ms, 3),
            "vram_before_mb": before_snapshot,
            "vram_after_mb": after_snapshot,
            "vram_delta_mb": self._subtract_vram_snapshot(after_snapshot, before_snapshot),
        }

        self._release_temporaries(
            text_latent=locals().get("text_latent"),
            result_tensor=locals().get("result_tensor"),
            latent_bald=locals().get("latent_bald"),
        )
        return payload

    # BiSeNet CelebAMask-HQ face parsing class indices to EXCLUDE from compositing.
    # These are all face-identity regions that must stay from the original photo.
    # 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 7=nose, 8=u_lip, 9=d_lip, 11=neck
    _FACE_IDENTITY_CLASSES: Tuple[int, ...] = (1, 2, 3, 4, 5, 7, 8, 9, 11)

    def _build_face_exclusion_mask(self, parsing_mask, device):
        """Build a binary mask of ALL face-identity regions that must NOT be overwritten.

        This is critical for preserving the original identity (especially for
        non-FFHQ-distribution faces such as Korean/Asian faces). The StyleGAN2
        FFHQ generator is biased toward Western features, so any bleed from
        the generated image into face-identity regions will change the person's
        appearance.
        """
        import torch

        face_mask = torch.zeros_like(parsing_mask, dtype=torch.float32, device=device)
        for cls_idx in self._FACE_IDENTITY_CLASSES:
            face_mask = torch.maximum(face_mask, (parsing_mask == cls_idx).float())
        return face_mask

    def _build_output_alpha_mask(
        self,
        runtime: RuntimeBundle,
        edited_result_tensor: Any,
        segmentation: SegmentationResult,
    ):
        import cv2
        import numpy as np
        import torch

        with torch.no_grad():
            with autocast_context(self.device, self.config.use_autocast):
                edited_logits_1024, _ = runtime.seg(edited_result_tensor.to(self.device))
            edited_parsing = torch.argmax(edited_logits_1024.float(), dim=1, keepdim=True).long()

        # Start with hair + ear from the EDITED (StyleGAN) output
        edited_mask = (
            (edited_parsing == self.config.hair_class_index)
            | (edited_parsing == EAR_CLASS_INDEX)
        ).float()

        # Build face-identity exclusion mask from the ORIGINAL source parsing.
        # This ensures we NEVER composite StyleGAN pixels over face regions,
        # preserving the original person's ethnicity, identity, and features.
        face_exclusion = self._build_face_exclusion_mask(
            segmentation.parsing_mask, edited_mask.device
        )

        # Also build a face-identity mask from the EDITED output,
        # to catch cases where the StyleGAN output has face pixels in
        # slightly different positions (e.g., reshaped forehead).
        edited_face_exclusion = self._build_face_exclusion_mask(
            edited_parsing, edited_mask.device
        )
        # Union of both exclusion masks for maximum safety
        combined_face_exclusion = torch.maximum(face_exclusion, edited_face_exclusion)

        # OUTPUT ALPHA:
        # edited_mask (BiSeNet on StyleGAN output) knows WHERE hair was generated.
        # But when CLIP optimization is aggressive, StyleGAN can generate hair that
        # extends far beyond the original hair boundary, causing ghost artifacts
        # (dark hair blob covering face/sky after warping to original photo space).
        #
        # Fix: constrain edited_mask to (dilated) ORIGINAL hair region in aligned
        # space, BEFORE warping. This prevents large generated-hair blobs from
        # bleeding outside the original hair zone while still allowing the style
        # texture/shape to change within the original hair area.
        # Dilation of 20px in 1024 space gives ~2% expansion for slight style change.
        orig_hair_np = (
            segmentation.hair_mask.squeeze().detach().cpu().float().numpy()
        )
        _dil_px = 20  # 20px in aligned 1024×1024 space ≈ 2% expansion
        _dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_dil_px, _dil_px))
        orig_hair_dilated_np = cv2.dilate(orig_hair_np, _dil_k, iterations=1)
        orig_hair_dilated_t = (
            torch.from_numpy(orig_hair_dilated_np)
            .to(edited_mask.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        constrained_edited = edited_mask * (1.0 - combined_face_exclusion) * orig_hair_dilated_t
        alpha_mask = constrained_edited
        log(
            f"[alpha] sam_refined={segmentation.sam_refined}  "
            f"edited={constrained_edited.mean().item():.3f}  "
            f"face_excl={combined_face_exclusion.mean().item():.3f}  "
            f"combined={alpha_mask.squeeze().float().mean().item():.3f}"
        )
        alpha_mask = alpha_mask.squeeze().detach().cpu().numpy().astype(np.float32)

        # Use EROSION first to shrink the mask inward, pulling it away from
        # face boundaries, then a small dilation for smooth edges.
        kernel = scaled_mask_morph_kernel(int(max(alpha_mask.shape)))
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        # Erode to pull mask away from face-hair boundary
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, kernel // 2), max(3, kernel // 2))
        )
        alpha_mask = cv2.erode(alpha_mask, erode_kernel, iterations=1)
        # Then dilate slightly less than the original erosion
        small_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, kernel // 3), max(3, kernel // 3))
        )
        alpha_mask = cv2.dilate(alpha_mask, small_dilate, iterations=1)
        # Smooth edges - sigma를 작게 유지해서 비헤어 영역(가슴 등)으로 bleeding 방지
        sigma = max(1.5, kernel / 6.0)
        alpha_mask = cv2.GaussianBlur(alpha_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # 매우 낮은 alpha 값 제거: 헤어 경계 밖의 ghosting/blur bleeding 차단
        alpha_mask = np.where(alpha_mask > 0.15, alpha_mask, 0.0)
        return np.clip(alpha_mask, 0.0, 1.0)

    def _color_match_hair(
        self,
        generated_rgb: "np.ndarray",
        original_rgb: "np.ndarray",
        hair_mask_binary: "np.ndarray",
    ) -> "np.ndarray":
        """생성된 hair 영역의 색상 통계를 원본 사진 조명에 맞춤 (LAB 공간).

        StyleGAN은 FFHQ 스튜디오 조명 기준으로 생성하므로, 야외 원본 사진과
        색상/밝기가 다를 수 있음. LAB mean/std transfer로 자연스럽게 맞춤.
        """
        import cv2
        import numpy as np

        if hair_mask_binary.sum() < 100:
            return generated_rgb  # hair 영역이 너무 작으면 스킵

        gen_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ori_lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        result_lab = gen_lab.copy()
        for ch in range(3):
            gen_vals = gen_lab[..., ch][hair_mask_binary > 0.5]
            ori_vals = ori_lab[..., ch][hair_mask_binary > 0.5]
            if gen_vals.std() < 1e-6:
                continue
            # mean/std transfer
            corrected = (gen_lab[..., ch] - gen_vals.mean()) / gen_vals.std() * ori_vals.std() + ori_vals.mean()
            # hair 영역에만 적용, 나머지는 원본 유지
            result_lab[..., ch] = np.where(hair_mask_binary > 0.5, corrected, gen_lab[..., ch])

        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

    def _composite_aligned_result_to_original(
        self,
        runtime: RuntimeBundle,
        original_rgb: Any,
        edited_aligned_rgb: Any,
        edited_result_tensor: Any,
        segmentation: SegmentationResult,
        alignment_transform: AlignmentTransform,
    ):
        import cv2
        import numpy as np

        alpha_mask = self._build_output_alpha_mask(runtime, edited_result_tensor, segmentation)
        width, height = alignment_transform.original_size

        warped_rgb = cv2.warpPerspective(
            edited_aligned_rgb,
            alignment_transform.aligned_to_original,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_alpha = cv2.warpPerspective(
            alpha_mask,
            alignment_transform.aligned_to_original,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        warped_alpha = cv2.GaussianBlur(warped_alpha, (0, 0), sigmaX=1.5, sigmaY=1.5)
        warped_alpha = np.clip(warped_alpha, 0.0, 1.0)
        binary_mask = (warped_alpha > 0.1).astype(np.uint8)  # Poisson/color match용 binary

        # ── 컬러 매칭: StyleGAN 조명 → 원본 조명으로 보정 ──────────────────
        if self.config.use_color_match and binary_mask.sum() > 0:
            try:
                warped_rgb = self._color_match_hair(warped_rgb, original_rgb, warped_alpha)
                log("[composite] color_match applied")
            except Exception as e:
                log(f"[composite] color_match failed (skipped): {e}")

        # ── Poisson seamless cloning: 경계 gradient 자연스럽게 이어줌 ────────
        if self.config.use_poisson_blend and binary_mask.sum() > 100:
            try:
                # OpenCV seamlessClone: BGR 필요
                src_bgr = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)
                dst_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
                # center: hair mask 무게중심
                moments = cv2.moments(binary_mask)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    cx = np.clip(cx, 1, width - 2)
                    cy = np.clip(cy, 1, height - 2)
                    poisson_mask = (binary_mask * 255).astype(np.uint8)
                    blended_bgr = cv2.seamlessClone(
                        src_bgr, dst_bgr, poisson_mask, (cx, cy), cv2.NORMAL_CLONE
                    )
                    result = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
                    log(f"[composite] poisson_blend applied  center=({cx},{cy})")
                    return result.astype(np.uint8)
            except Exception as e:
                log(f"[composite] poisson_blend failed (fallback to alpha): {e}")

        # ── fallback: 일반 alpha blending ────────────────────────────────────
        warped_alpha_3ch = warped_alpha[..., None]
        composite = (
            warped_rgb.astype(np.float32) * warped_alpha_3ch
            + original_rgb.astype(np.float32) * (1.0 - warped_alpha_3ch)
        )
        return np.clip(composite, 0, 255).astype(np.uint8)

    def _get_cached_or_invert(self, runtime: RuntimeBundle, aligned_path: str):
        if self.config.preset_config.use_latent_cache:
            cached = self.latent_cache.get(aligned_path, self.config.preset)
            if cached is not None:
                log("[Cache] latent cache hit.")
                return cached[0].to(self.device), cached[1].to(self.device)
        src_latent, latent_f = runtime.embedding.invert_image_in_fs(aligned_path)
        if self.config.preset_config.use_latent_cache:
            self.latent_cache.set(aligned_path, self.config.preset, src_latent, latent_f)
        return src_latent, latent_f

    def _request_output_dir(self, image_path: str, output_dir: Optional[str]) -> Path:
        if output_dir:
            directory = Path(output_dir)
        else:
            request_id = hashlib.sha256(
                f"{image_path}:{time.time_ns()}:{self.config.preset}".encode("utf-8")
            ).hexdigest()[:12]
            directory = self.config.result_dir / request_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _aligned_cache_path(self, image_path: str) -> str:
        path = Path(image_path)
        stat = path.stat()
        digest = hashlib.sha256(
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{self.config.target_output_size}".encode("utf-8")
        ).hexdigest()
        return str(self.config.aligned_cache_dir / f"{digest}.png")

    def _align_and_cache(self, image_path: str, image_bgr: Any, observation: FaceObservation) -> Tuple[str, Any, AlignmentTransform]:
        aligned_path = self._aligned_cache_path(image_path)
        transform = self.aligner.build_transform(image_bgr, observation)
        if Path(aligned_path).is_file():
            return aligned_path, self._load_image_rgb(aligned_path), transform
        alignment_result = self.aligner.align(image_bgr, observation, transform=transform)
        alignment_result.aligned_image.save(aligned_path)
        return aligned_path, alignment_result.aligned_rgb, transform

    def _vectorized_image_tensor(self, image_input: Any):
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as F

        if isinstance(image_input, (str, Path)):
            image_bgr = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Unable to read image: {image_input}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            image_rgb = image_input
        else:
            raise TypeError(f"Unsupported image input: {type(image_input)}")
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=self.device.type == "cuda") / 255.0
        tensor = F.interpolate(
            tensor,
            size=(self.config.target_output_size, self.config.target_output_size),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.mul(2.0).sub(1.0)

    @staticmethod
    def _load_image_bgr(image_input: Any):
        import cv2
        import numpy as np
        import PIL.Image

        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unable to load image: {image_input}")
            return image
        if isinstance(image_input, PIL.Image.Image):
            return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        if isinstance(image_input, np.ndarray):
            return image_input
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    @staticmethod
    def _bgr_to_rgb(image_bgr):
        import cv2

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_image_rgb(image_path: str):
        import cv2

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unable to read image: {image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _current_vram_snapshot(self) -> Dict[str, float]:
        import torch

        if self.device.type != "cuda":
            return {}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated(self.device) / (1024 ** 2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(self.device) / (1024 ** 2), 2),
        }

    @staticmethod
    def _subtract_vram_snapshot(after: Mapping[str, float], before: Mapping[str, float]) -> Dict[str, float]:
        delta: Dict[str, float] = {}
        for key in set(after) | set(before):
            delta[key] = round(float(after.get(key, 0.0)) - float(before.get(key, 0.0)), 2)
        return delta

    def _release_temporaries(self, **kwargs: Any) -> None:
        del kwargs
        gc.collect()
        if self.device.type == "cuda" and self.config.empty_cache_per_task:
            import torch

            torch.cuda.empty_cache()

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value)
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_") or "style"


HairCLIPv2ServicePipeline = MirrAIOptimizedPipeline


def generate_multiple_styles(
    image_path: str,
    text_prompt: str = "",
    config: Optional[ServiceConfig] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    pipeline = MirrAIOptimizedPipeline(config=config)
    return pipeline.generate_multiple_styles(image_path=image_path, text_prompt=text_prompt, **kwargs)


def build_config_from_args(args) -> ServiceConfig:
    return ServiceConfig(
        device=args.device,
        preset=args.preset,
        precision=args.precision,
        enable_tensorrt=args.enable_tensorrt,
        enable_torch_compile=args.enable_torch_compile,
        async_load=not args.disable_async_load,
        trend_remote_url=args.trend_remote_url or os.environ.get("HAIR_TREND_REMOTE_URL"),
        trend_refresh_seconds=args.trend_refresh_seconds,
        trend_limit=args.trend_limit,
        locale=args.locale,
        alignment_intermediate_size=args.alignment_intermediate_size,
        alignment_upscale_min_size=args.alignment_upscale_min_size,
        alignment_max_upscale_factor=args.alignment_max_upscale_factor,
        enable_sam2=args.enable_sam2,
        sam2_checkpoint_path=args.sam2_checkpoint,
        sam2_auto_download=not args.disable_sam2_auto_download,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MirrAI optimized HairCLIPv2 pipeline")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--text", default="", help="Target hairstyle prompt")
    parser.add_argument("--output-dir", default=None, help="Output directory for generated images")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--preset", default="realtime", choices=sorted(PRESETS.keys()))
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--locale", default="global", help="Trend locale filter")
    parser.add_argument("--trend-limit", default=5, type=int, help="Recommendation count")
    parser.add_argument("--trend-remote-url", default=None, help="Remote trend JSON endpoint")
    parser.add_argument("--trend-refresh-seconds", default=1800, type=int, help="Remote trend cache TTL")
    parser.add_argument("--skip-align", action="store_true", help="Skip face alignment")
    parser.add_argument(
        "--alignment-intermediate-size",
        default=GENERATOR_OUTPUT_SIZE * 2,
        type=int,
        help="Intermediate resolution used before the final aligned resize",
    )
    parser.add_argument(
        "--alignment-upscale-min-size",
        default=GENERATOR_OUTPUT_SIZE,
        type=int,
        help="Upscale small inputs before alignment until the shorter side reaches this size",
    )
    parser.add_argument(
        "--alignment-max-upscale-factor",
        default=2.0,
        type=float,
        help="Maximum pre-alignment upscale factor for small inputs",
    )
    parser.add_argument("--recommend-only", action="store_true", help="Return analysis and recommendations only")
    parser.add_argument("--prepare-only", action="store_true", help="Precompute aligned identity cache only")
    parser.add_argument("--return-base64", action="store_true", help="Include base64 image payloads")
    parser.add_argument("--enable-tensorrt", action="store_true", help="Try TensorRT for segmentation")
    parser.add_argument("--enable-torch-compile", action="store_true", help="Try torch.compile for generator")
    parser.add_argument("--enable-sam2", action="store_true", help="Enable optional SAM2-guided masking refinement")
    parser.add_argument("--sam2-checkpoint", default=None, help="Path to a local SAM2 checkpoint")
    parser.add_argument(
        "--disable-sam2-auto-download",
        action="store_true",
        help="Disable automatic SAM2 checkpoint download via Hugging Face",
    )
    parser.add_argument("--disable-async-load", action="store_true", help="Disable async runtime warmup")
    args = parser.parse_args()

    config = build_config_from_args(args)
    pipeline = MirrAIOptimizedPipeline(config=config)
    try:
        if args.recommend_only:
            analysis = pipeline.analyze_face(args.image)
            payload = {
                "face_analysis": analysis,
                "recommendation": pipeline.recommend_styles(analysis, prompt=args.text),
                "eta": pipeline.estimate_eta(args.image, text_prompt=args.text, skip_alignment=args.skip_align),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        if args.prepare_only:
            payload = pipeline.prepare_identity(args.image, skip_alignment=args.skip_align)
            payload["eta"] = pipeline.estimate_eta(args.image, text_prompt=args.text, skip_alignment=args.skip_align)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        payload = pipeline.generate_multiple_styles(
            image_path=args.image,
            text_prompt=args.text,
            skip_alignment=args.skip_align,
            return_base64=args.return_base64,
            output_dir=args.output_dir,
            limit=args.trend_limit,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
