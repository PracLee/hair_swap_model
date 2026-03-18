"""
MirrAI SD Inpainting Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

완전 생성형(Text→Hair) 파이프라인.

아키텍처:
  ┌─────────────────────────────────────────────────────────────┐
  │  입력: 사진 + hairstyle_text + color_text                    │
  ├─────────────────────────────────────────────────────────────┤
  │  [1] MediaPipe FaceDetection → 얼굴 bbox + landmarks        │
  │  [2] BiSeNet → base hair mask (원본 해상도)                  │
  │  [3] SAM2   → 정밀 hair mask 보정 (point + text prompt)     │
  │  [4] Canny edge → ControlNet conditioning (얼굴 구조 보존)  │
  │  [5] face crop → IP-Adapter conditioning (얼굴 identity)    │
  │  [6] SD 1.5 Inpainting + ControlNet → hair 영역 생성        │
  │  [7] Composite → 원본 얼굴 유지 + 생성 헤어 합성             │
  └─────────────────────────────────────────────────────────────┘
  출력: top-k 결과 이미지 (각기 다른 seed)

모델:
  - BiSeNet: pretrained_models/seg.pth (기존 모델 재사용)
  - SAM2:    pretrained_models/sam2.pt  (기존 모델 재사용)
  - SD Inpaint: runwayml/stable-diffusion-inpainting (HF Hub)
  - ControlNet: lllyasviel/control_v11p_sd15_canny   (HF Hub)
  - IP-Adapter: h94/IP-Adapter / ip-adapter-plus-face_sd15.bin (HF Hub)
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# ── HuggingFace 모델 ID ────────────────────────────────────────────────────────
SD_INPAINT_MODEL_ID   = "runwayml/stable-diffusion-inpainting"
CONTROLNET_MODEL_ID   = "lllyasviel/control_v11p_sd15_canny"
IP_ADAPTER_REPO_ID    = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT     = "ip-adapter-plus-face_sd15.bin"

# ── SegFace 설정 ───────────────────────────────────────────────────────────────
HAIR_CLASS_IDX   = 14
# 0: bg, 1: neck, 2: face, 3: cloth, 4: r_ear, 5: l_ear, 6: r_bro, 7: l_bro, 
# 8: r_eye, 9: l_eye, 10: nose, 11: inner_mouth, 12: lower_lip, 13: upper_lip
# 얼굴 내부 및 목/귀 클래스 포함
FACE_CLASS_IDXS  = frozenset([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18])
# 17: earring, 18: necklace → SD가 귀걸이/목걸이 임의 생성하는 문제 방지
CLOTH_CLASS_IDX  = 3   # SegFace class 3 = cloth → hair mask에서 제거해 옷 영역 보호

# ── SD 생성 해상도 ─────────────────────────────────────────────────────────────
SD_SIZE = 512   # SD 1.5 native resolution

# ── 공통 네거티브 프롬프트 ─────────────────────────────────────────────────────
_NEGATIVE_BASE = (
    "ugly, deformed, blurry, low quality, bad anatomy, distorted face, "
    "distorted hair, bald patch, artifacts, watermark, signature, "
    "cartoon, anime, illustration, painting, drawing, "
    "earrings, earring, dangling earrings, hoop earrings, pearl earrings, "
    "jewelry, necklace, pendant, choker, accessories, piercings, ear accessories"
)

# ── 헤어 길이 키워드 ────────────────────────────────────────────────────────────
_SHORT_HAIR_KEYWORDS = frozenset([
    "short", "bob", "pixie", "buzz", "hush", "crop", "cropped",
    "undercut", "bowl", "chin length", "chin-length",
    "above ear", "above shoulder", "ear length", "single",
    "단발", "숏컷", "픽시",
])
_MEDIUM_HAIR_KEYWORDS = frozenset([
    "lob", "midi", "medium", "shoulder length", "shoulder-length",
    "collarbone", "clavicle", "mid length", "mid-length",
])

_NO_COLOR_HINTS = frozenset([
    "", "none", "no color", "same", "original", "default",
    "원본", "기존", "유지", "없음",
])

# RGB 기준 타겟 컬러 (근사값)
_HAIR_COLOR_TARGET_RGB: List[Tuple[str, Tuple[int, int, int]]] = [
    ("ash beige", (173, 158, 136)),
    ("ash brown", (111, 92, 80)),
    ("ash blonde", (192, 176, 146)),
    ("ash black", (58, 58, 62)),
    ("ash gray", (124, 128, 134)),
    ("ash grey", (124, 128, 134)),
    ("ash", (128, 126, 124)),
    ("black", (44, 41, 39)),
    ("dark brown", (82, 62, 50)),
    ("brown", (98, 74, 58)),
    ("beige", (174, 153, 128)),
    ("blonde", (193, 166, 121)),
    ("silver", (170, 174, 182)),
    ("gray", (132, 132, 132)),
    ("grey", (132, 132, 132)),
    ("red", (128, 56, 45)),
    ("auburn", (120, 63, 48)),
    ("pink", (170, 112, 132)),
    ("blue", (82, 95, 138)),
]

_BG_FILL_MODE_ALIASES = {
    "lama": "lama",
    "sd": "sd",
    "cv2": "lama",  # legacy alias
}


def normalize_bg_fill_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    normalized = _BG_FILL_MODE_ALIASES.get(mode)
    if normalized is None:
        raise ValueError("bg_fill_mode must be one of: lama, sd")
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SDInpaintConfig:
    """SD Inpainting 파이프라인 설정"""
    # SD 생성 파라미터
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 0.3   # 낮춰야 텍스트 프롬프트가 먹힘
    ip_adapter_scale: float = 0.35               # 너무 강하면 원본 헤어 유지해버림

    # Canny edge 파라미터
    canny_low: int  = 80
    canny_high: int = 200

    # hair mask dilate (SD 입력용 — 경계 확장, 잔머리 커버용으로 넉넉하게)
    # 얼굴 내부 보호는 BiSeNet face_region_mask 로 픽셀 단위 처리함
    mask_dilate_px: int = 30

    # IP-Adapter 얼굴 crop padding 비율
    face_crop_padding: float = 0.25

    # 씨드 리스트 — None 이면 요청마다 랜덤 생성 (권장), 고정값 지정도 가능
    seeds: Optional[List[int]] = None

    # 디바이스 / dtype
    device: str = "cuda"
    dtype: str = "float16"

    # SAM2 사용 여부
    use_sam2: bool = True

    # Optional: custom hair-only SegFace fine-tune (HF repo)
    segface_hair_repo_id: str = os.environ.get("SEGFACE_HAIR_REPO_ID", "").strip()
    segface_hair_revision: str = os.environ.get("SEGFACE_HAIR_REVISION", "main").strip()

    # 메모리 최적화
    enable_xformers: bool = False

    # 후처리 옵션 (현재 파이프라인에서는 기본 alpha blend 사용)
    use_clip_ranking: bool = False   # 향후 CLIP 랭킹 확장용
    use_color_match:  bool = False   # 향후 LAB 색상 매칭 확장용
    use_poisson_blend: bool = False  # 향후 Poisson blend 확장용

    # 하이브리드 pre-clean 모드 (short/medium 변환 시 하단 긴머리 선철거 방법)
    #   "lama": LaMa partial pre-clean만 수행
    #   "sd"  : LaMa partial pre-clean 후 SD refinement까지 수행
    #   legacy alias "cv2"는 "lama"로 정규화된다.
    bg_fill_mode: str = "lama"

    # short/medium 2-step 전략:
    #   step-1: long hair 흔적 제거(pre-clean, tied/slicked back 컨셉)
    #   step-2: target hairstyle 생성
    enable_two_step_preclean: bool = True
    preclean_mask_expand_ratio_x: float = 1.65
    preclean_mask_expand_ratio_y: float = 1.00
    preclean_strength: float = 0.96

    def __post_init__(self) -> None:
        self.bg_fill_mode = normalize_bg_fill_mode(self.bg_fill_mode)


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SDInpaintResult:
    image: np.ndarray       # H×W×3 BGR (원본 해상도)
    image_pil: Image.Image  # PIL RGB
    seed: int
    rank: int
    mask_used: str          # "sam2" | "bisenet"
    clip_score: float = 0.0 # legacy exported ranking score slot
    color_score: float = 0.0
    silhouette_score: float = 0.0
    rank_score: float = 0.0
    mask: Optional[np.ndarray] = None       # H×W float32 디버그용 마스크
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    debug_images: Optional[Dict[str, np.ndarray]] = None    # 디버그용 중간 산출물 (BGR)
    debug_data: Optional[Dict[str, Any]] = None             # 디버그용 중간 메타데이터(JSON)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MirrAISDPipeline:
    """
    SAM2 + SD Inpainting + ControlNet(Canny) + IP-Adapter 기반 헤어 변환 파이프라인.

    - hair segmentation: 기존 BiSeNet seg.pth + SAM2 sam2.pt 재사용
    - 생성:              SD 1.5 Inpainting + ControlNet canny + IP-Adapter face
    """

    def __init__(self, config: Optional[SDInpaintConfig] = None) -> None:
        self.config = config or SDInpaintConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.dtype = (
            torch.float16 if self.config.dtype == "float16" else torch.bfloat16
        )

        self._segface    = None   # SegFace (Swin-B) face parsing
        self._segface_hair = None # Optional: custom SegFace hair-only model
        self._segface_hair_threshold = 0.5
        self._sam2_factory = None  # SAM2 predictor factory (callable)
        self._sd_pipe    = None   # StableDiffusionControlNetInpaintPipeline
        self._mp_face    = None   # MediaPipe FaceDetection
        self._mp_face_mesh = None # MediaPipe FaceMesh
        self._lama       = None   # LaMa large mask inpainting
        self._loaded     = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """모델 로드 (cold start). 이미 로드된 경우 no-op."""
        if self._loaded:
            return
        logger.info("[SDPipeline] 모델 로딩 시작...")
        self._load_segface()
        self._load_segface_hair()
        self._load_sam2()
        self._load_mediapipe()
        self._load_sd_pipeline()
        self._load_lama()
        self._loaded = True
        logger.info("[SDPipeline] 모든 모델 로드 완료")

    def run(
        self,
        image: np.ndarray,     # BGR, any resolution
        hairstyle_text: str,
        color_text: str,
        top_k: int = 3,
        return_intermediates: bool = False,
    ) -> List[SDInpaintResult]:
        """
        헤어 스타일 변환 실행.

        Args:
            image:          입력 이미지 (BGR numpy)
            hairstyle_text: 헤어스타일 텍스트 (트렌드 데이터 hairstyle_text)
            color_text:     헤어 컬러 텍스트 (트렌드 데이터 color_text)
            top_k:          반환 결과 수 (기본 3)
            return_intermediates: 중간 산출물 디버그 이미지 포함 여부

        Returns:
            SDInpaintResult 리스트 (rank 0이 first)
        """
        if not self._loaded:
            self.load()

        # 시드 결정: config에 고정값 있으면 사용, 없으면 매 요청마다 랜덤 생성
        if self.config.seeds:
            seeds = self.config.seeds[:top_k]
        else:
            seeds = [random.randint(0, 2**31 - 1) for _ in range(top_k)]
        logger.info(f"[SDPipeline] seeds={seeds}")

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bg_fill_mode = normalize_bg_fill_mode(self.config.bg_fill_mode)
        self.config.bg_fill_mode = bg_fill_mode
        normalized_color_text = self._normalize_color_text(color_text)
        has_color_request = bool(normalized_color_text)
        target_hair_lab = self._resolve_target_hair_lab(normalized_color_text) if has_color_request else None
        if not has_color_request:
            logger.info("[SDPipeline] color_text 미지정 → 원본 머리 톤 유지 모드")
        elif target_hair_lab is None:
            logger.info("[SDPipeline] color_text 파싱 실패 → 색상 재정렬은 스킵")

        H, W = image.shape[:2]
        debug_images_common: Optional[Dict[str, np.ndarray]] = {} if return_intermediates else None
        debug_data_common: Optional[Dict[str, Any]] = {} if return_intermediates else None

        def _store_mask(name: str, mask: np.ndarray) -> None:
            if debug_images_common is None:
                return
            m = np.clip(mask, 0.0, 1.0)
            m_u8 = (m * 255).astype(np.uint8)
            debug_images_common[name] = cv2.cvtColor(m_u8, cv2.COLOR_GRAY2BGR)

        def _store_rgb(name: str, rgb_img: np.ndarray) -> None:
            if debug_images_common is None:
                return
            debug_images_common[name] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        def _store_debug_bundle(bundle: Optional[Dict[str, Any]]) -> None:
            if not bundle:
                return
            for name, value in bundle.items():
                if value is None:
                    continue
                if isinstance(value, np.ndarray):
                    if value.ndim == 2:
                        _store_mask(name, value.astype(np.float32))
                    elif value.ndim == 3 and value.shape[2] == 3:
                        _store_rgb(name, value.astype(np.uint8))
                elif debug_data_common is not None and isinstance(value, (bool, int, float, str)):
                    debug_data_common[name] = value

        if debug_images_common is not None:
            debug_images_common["pipeline_input_image"] = image.copy()
        if debug_data_common is not None:
            debug_data_common["hair_mask_model"] = (
                self.config.segface_hair_repo_id if self._segface_hair is not None else "segface_default"
            )
            debug_data_common["bg_fill_mode"] = bg_fill_mode

        # ── Step 1: 얼굴 검출 ────────────────────────────────────────────────
        face_obs = self._detect_face(img_rgb)
        if face_obs is None:
            raise ValueError("얼굴을 검출할 수 없습니다.")
        face_bbox = face_obs  # (x1, y1, x2, y2)
        logger.info(f"[SDPipeline] 얼굴 검출: {face_bbox}")
        mesh_norm, mesh_px = self._detect_face_mesh(img_rgb)
        if mesh_norm is not None and mesh_px is not None:
            logger.info(f"[SDPipeline] FaceMesh 검출: landmarks={len(mesh_norm)}")
            if debug_images_common is not None:
                mesh_images = self._render_face_mesh_debug_images(img_rgb, mesh_px)
                debug_images_common.update(mesh_images)
            if debug_data_common is not None:
                debug_data_common["mediapipe_face_mesh"] = self._build_face_mesh_analysis(
                    mesh_norm, mesh_px
                )
        elif debug_data_common is not None:
            debug_data_common["mediapipe_face_mesh"] = {"detected": False}

        # ── Step 2: SegFace base hair mask + 얼굴 픽셀 마스크 + 옷 마스크 ──────
        hair_mask_base, face_region_mask, cloth_mask = self._segface_hair_mask(img_rgb, face_bbox)
        _store_mask("segface_hair_mask", hair_mask_base)
        _store_mask("segface_face_region_mask", face_region_mask)
        _store_mask("segface_cloth_mask", cloth_mask)
        feature_protect_mask = self._build_feature_protect_mask(
            face_region_mask,
            face_bbox=face_bbox,
            landmarks_px=mesh_px,
        )
        _store_mask("pipeline_face_feature_protect_mask", feature_protect_mask)

        # ── Step 3: SAM2 refinement ───────────────────────────────────────────
        hair_mask, mask_source = self._refine_with_sam2(
            img_rgb, hair_mask_base, face_bbox, hairstyle_text
        )
        logger.info(
            f"[SDPipeline] hair mask source={mask_source}, "
            f"pixels={hair_mask.sum():.0f}"
        )
        _store_mask(f"{mask_source}_refined_hair_mask", hair_mask)

        if hair_mask.sum() < 300:
            raise ValueError("머리카락 영역이 너무 작습니다.")

        # ── Step 3-b: 헤어 길이 분류 ─────────────────────────────────────────
        hair_length = self._classify_hair_length(hairstyle_text)
        logger.info(f"[SDPipeline] 헤어 길이 분류: {hair_length}")

        # ── Step 3-c: SegFace 얼굴 픽셀 제거 (bbox 직사각형 대신 픽셀 단위 보정) ─
        hair_mask = np.clip(hair_mask - feature_protect_mask, 0.0, 1.0)
        # short/medium 긴머리 제거 단계에서는 "옷 위로 떨어진 머리카락"도 지워야 하므로
        # cloth 제거 전 마스크를 별도로 보관한다.
        hair_mask_for_removal = hair_mask.copy()
        _store_mask("pipeline_hair_mask_face_protected", hair_mask_for_removal)
        logger.info(
            f"[SDPipeline] 얼굴 픽셀 제거 완료, pixels={hair_mask.sum():.0f}"
        )

        # ── Step 3-d: SegFace 옷 픽셀 제거 (옷이 바뀌는 문제 방지) ────────────
        # expand 전에 먼저 제거해야 옷 영역이 마스크 확장에 영향받지 않음
        cloth_mask_dilated = self._dilate_mask(cloth_mask)
        hair_mask = np.clip(hair_mask - cloth_mask_dilated, 0.0, 1.0)
        _store_mask("segface_cloth_mask_dilated", cloth_mask_dilated)
        _store_mask("pipeline_hair_mask_cloth_protected", hair_mask)
        logger.info(
            f"[SDPipeline] 옷 픽셀 제거 완료, pixels={hair_mask.sum():.0f}"
        )

        # ── Step 3-e: 숏컷/중단발 — 하이브리드 전략 ───────────────────────────
        # 하단 긴머리를 먼저 부분 선철거한 뒤 새 short를 생성하고,
        # 마지막에 잔여 long-hair만 차집합 마스크로 정리한다.
        cutoff_y_for_post: Optional[int] = None
        long_hair_mask_for_post: Optional[np.ndarray] = None
        shoulder_protect_for_post: Optional[np.ndarray] = None
        preclean_mask_for_input: Optional[np.ndarray] = None
        if hair_length in ("short", "medium"):
            x1f, y1f, x2f, y2f = face_bbox
            face_w = max(x2f - x1f, 1)
            face_h = max(y2f - y1f, 1)

            if hair_length == "short":
                cutoff_y = int(y2f + face_h * 0.02)   # 턱선 바로 아래
            else:
                cutoff_y = int(y2f + face_h * 0.54)   # 어깨 위 (끝선 명확도 강화)
            cutoff_y = min(cutoff_y, H - 1)
            cutoff_y_for_post = cutoff_y
            shoulder_protect_for_post = self._build_shoulder_protect_mask(
                cloth_mask=cloth_mask_dilated,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
            )
            if shoulder_protect_for_post.sum() > 0:
                logger.info(
                    "[SDPipeline] 어깨 보호 마스크 적용: "
                    f"pixels={shoulder_protect_for_post.sum():.0f}"
                )
            _store_mask("pipeline_shoulder_protect_mask", shoulder_protect_for_post)

            # ── head box 계산 (gen_mask corridor 범위 산정에 사용) ───────────────
            margin_x = int(face_w * 0.5)   # 얼굴 폭의 50% 여백 (양 옆 머리 공간)
            head_x1  = max(0, x1f - margin_x)
            head_x2  = min(W, x2f + margin_x)
            head_y1  = max(0, y1f - int(face_h * 0.6))   # 정수리 위까지
            head_y2  = cutoff_y

            # 기존 긴 머리 전체 마스크.
            # post cleanup에서는 실제 생성 결과의 short mask와 차집합을 다시 계산한다.
            long_hair_mask = np.clip(hair_mask_for_removal - feature_protect_mask, 0.0, 1.0)
            long_hair_mask_for_post = long_hair_mask.copy()

            # gen_mask = 기존 헤어 상단 + 얼굴 주변 corridor를 합쳐 새 단발이 생성될 영역만 열어 준다.
            # 직사각형 전체 head box를 쓰면 배경 패치가 같이 생겨 사각형 artifact가 남기 쉽다.
            gen_mask = long_hair_mask.copy()
            gen_soft_bottom = min(
                H,
                cutoff_y + max(8, int(face_h * (0.03 if hair_length == "short" else 0.16))),
            )
            gen_mask[gen_soft_bottom:, :] = 0.0

            # 직사각 corridor 대신 타원형 head prior를 더해 사각형 artifact를 줄인다.
            corridor_y_pad = max(8, int(face_h * (0.03 if hair_length == "short" else 0.22)))
            cx = int(0.5 * (x1f + x2f))
            cy = int(y1f + face_h * (0.34 if hair_length == "short" else 0.40))
            ellipse_axes = (
                max(24, int(face_w * (0.78 if hair_length == "short" else 1.00))),
                max(26, int(face_h * (0.78 if hair_length == "short" else 1.10))),
            )
            ellipse_u8 = np.zeros((H, W), dtype=np.uint8)
            cv2.ellipse(ellipse_u8, (cx, cy), ellipse_axes, 0, 0, 360, 255, -1)
            ellipse_mask = (ellipse_u8 > 0).astype(np.float32)
            ellipse_mask[min(H, cutoff_y + corridor_y_pad):, :] = 0.0
            gen_mask = np.clip(np.maximum(gen_mask, ellipse_mask), 0.0, 1.0)

            gen_kx = max(11, int(face_w * (0.10 if hair_length == "short" else 0.13)))
            gen_ky = max(11, int(face_h * (0.08 if hair_length == "short" else 0.12)))
            if gen_kx % 2 == 0:
                gen_kx += 1
            if gen_ky % 2 == 0:
                gen_ky += 1
            gen_u8 = (gen_mask > 0.30).astype(np.uint8) * 255
            gen_u8 = cv2.dilate(
                gen_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gen_kx, gen_ky)),
                iterations=1,
            )
            gen_u8 = cv2.morphologyEx(
                gen_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            gen_mask_base = (gen_u8 > 0).astype(np.float32)

            # short/medium에서는 턱 아래 남아 있는 기존 long-hair 영역도
            # SD가 다시 쓰도록 generation mask에 포함시킨다.
            # 이 확장이 없으면 preclean에서 남은 흐린 hair remnant가
            # composite 이후 그대로 살아남기 쉽다.
            expanded_lower_mask = self._expand_mask_for_short_hair(
                long_hair_mask.copy(),
                face_bbox=face_bbox,
                H=H,
                W=W,
                hair_length=hair_length,
            )
            lower_rewrite_mask = np.zeros((H, W), dtype=np.float32)
            if expanded_lower_mask.shape == (H, W):
                lower_rewrite_mask = np.clip(expanded_lower_mask.astype(np.float32), 0.0, 1.0)

            gen_mask_base = np.clip(gen_mask_base - feature_protect_mask, 0.0, 1.0)
            lower_rewrite_mask = np.clip(lower_rewrite_mask - feature_protect_mask, 0.0, 1.0)
            if shoulder_protect_for_post is not None and shoulder_protect_for_post.shape == (H, W):
                # 단발 silhouette 생성 영역은 기존 보호 강도를 유지하되,
                # lower rewrite mask는 더 약하게 보호해서 긴 스트랜드 제거를 우선한다.
                gen_mask_base = np.clip(gen_mask_base - (shoulder_protect_for_post * 0.55), 0.0, 1.0)
                lower_rewrite_mask = np.clip(lower_rewrite_mask - (shoulder_protect_for_post * 0.18), 0.0, 1.0)

            gen_mask = np.clip(np.maximum(gen_mask_base, lower_rewrite_mask), 0.0, 1.0)

            if debug_data_common is not None:
                debug_data_common["short_generation_mask_stats"] = {
                    "base_pixels": int((gen_mask_base > 0.35).sum()),
                    "lower_rewrite_pixels": int((lower_rewrite_mask > 0.35).sum()),
                    "final_pixels": int((gen_mask > 0.35).sum()),
                }

            _store_mask("pipeline_short_generation_mask", gen_mask)

            preclean_seed_mask = long_hair_mask.copy()
            preclean_top = max(0, int(cutoff_y - face_h * 0.03))
            preclean_seed_mask[:preclean_top, :] = 0.0
            preclean_mask_for_input = preclean_seed_mask.copy()
            if self.config.enable_two_step_preclean:
                preclean_mask_for_input = self._build_preclean_mask_for_two_step(
                    preclean_seed_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    cloth_mask=cloth_mask_dilated,
                    hair_length=hair_length,
                )

            logger.info(
                f"[SDPipeline] hybrid: long_px={long_hair_mask.sum():.0f}, "
                f"gen_px={gen_mask.sum():.0f}, preclean_px={preclean_mask_for_input.sum():.0f}, "
                f"cutoff_y={cutoff_y}, head_box=({head_x1},{head_y1})-({head_x2},{head_y2})"
            )

            img_rgb_cleaned = img_rgb
            _store_mask("pipeline_short_preclean_mask", preclean_mask_for_input)
            preclean_pixels = int((np.clip(preclean_mask_for_input, 0.0, 1.0) > 0.35).sum())
            if preclean_pixels >= 80:
                preclean_seed = int(seeds[0]) ^ 0x5A5A5A5A
                preclean_seed &= 0x7FFFFFFF
                face_crop_preclean = self._crop_face(Image.fromarray(img_rgb), face_bbox)
                preclean_debug: Dict[str, Any] = {
                    "pipeline_lama_preclean_mask": (np.clip(preclean_mask_for_input, 0.0, 1.0) > 0.35).astype(np.float32),
                    "lama_preclean_pixels": preclean_pixels,
                }
                try:
                    img_rgb_preclean = self._lama_inpaint(
                        img_rgb,
                        (np.clip(preclean_mask_for_input, 0.0, 1.0) > 0.35).astype(np.uint8) * 255,
                        force_single_pass=True,
                    )
                    preclean_debug["lama_preclean_method"] = "lama"
                    preclean_debug["pipeline_lama_preclean_result"] = img_rgb_preclean
                except Exception as e:
                    logger.warning(f"[SDPipeline] hybrid LaMa preclean 실패 → cv2 fallback: {e}")
                    img_rgb_preclean = self._cv2_inpaint_region(
                        img_rgb,
                        preclean_mask_for_input,
                        protect_mask=feature_protect_mask,
                    )
                    preclean_debug["lama_preclean_method"] = "cv2_fallback"
                    preclean_debug["pipeline_cv2_preclean_result"] = img_rgb_preclean
                _store_debug_bundle(preclean_debug)
                if bg_fill_mode == "sd":
                    img_rgb_cleaned = self._sd_preclean_long_hair_region(
                        img_rgb_preclean,
                        preclean_mask_for_input,
                        face_bbox=face_bbox,
                        face_crop_pil=face_crop_preclean,
                        protect_mask=feature_protect_mask,
                        hair_length=hair_length,
                        seed=preclean_seed,
                    )
                else:
                    img_rgb_cleaned = img_rgb_preclean

                preclean_residual_cleanup = self._remove_residual_hair_below_cutoff(
                    img_rgb_cleaned,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    shoulder_protect=shoulder_protect_for_post,
                    hair_length=hair_length,
                    return_debug=debug_images_common is not None,
                    debug_prefix="preclean_residual",
                    min_pixels=24 if hair_length == "short" else 48,
                    open_kernel_size=3 if hair_length == "short" else 5,
                    dilate_kernel_size=15 if hair_length == "short" else 9,
                    top_offset_px=-(max(8, int(face_h * 0.06))) if hair_length == "short" else 0,
                )
                if debug_images_common is not None:
                    img_rgb_cleaned, preclean_residual_debug = preclean_residual_cleanup
                    _store_debug_bundle(preclean_residual_debug)
                else:
                    img_rgb_cleaned = preclean_residual_cleanup

                logger.info(
                    f"[SDPipeline] hybrid preclean 완료: mode={bg_fill_mode}, "
                    f"pixels={preclean_pixels}"
                )

            _store_rgb("pipeline_background_cleaned_rgb", img_rgb_cleaned)

            hair_mask_for_sd = gen_mask.astype(np.float32)
            img_rgb_for_sd   = img_rgb_cleaned
        else:
            # long 헤어는 기존 단일 패스 유지
            hair_mask_for_sd = hair_mask
            img_rgb_for_sd   = img_rgb
            img_rgb_cleaned  = img_rgb

        _store_mask("sd_inpaint_mask", hair_mask_for_sd)
        _store_rgb("sd_input_rgb", img_rgb_for_sd)

        # ── Step 4: SD 입력 준비 ─────────────────────────────────────────────
        img_pil = Image.fromarray(img_rgb_cleaned)
        # short/medium에서는 기존 long-hair 윤곽도 억제해 ControlNet이
        # 원본 긴머리 edge를 새 단발 형상으로 따라가지 않게 한다.
        canny_suppress = None
        if hair_length in ("short", "medium"):
            canny_suppress = np.clip(
                np.maximum(
                    np.clip(hair_mask_for_removal, 0.0, 1.0),
                    np.clip(hair_mask_for_sd, 0.0, 1.0),
                ),
                0.0,
                1.0,
            )
        img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
            img_rgb_for_sd, hair_mask_for_sd,
            canny_suppress_mask=canny_suppress,
        )
        if debug_images_common is not None:
            debug_images_common["sd_input_512"] = cv2.cvtColor(
                np.array(img_512), cv2.COLOR_RGB2BGR
            )
            debug_images_common["sd_inpaint_mask_512"] = cv2.cvtColor(
                np.array(mask_512).astype(np.uint8), cv2.COLOR_GRAY2BGR
            )
            debug_images_common["controlnet_canny_512"] = cv2.cvtColor(
                np.array(canny_512), cv2.COLOR_RGB2BGR
            )

        # ── Step 5: 얼굴 crop (IP-Adapter) ───────────────────────────────────
        face_crop_pil = self._crop_face(img_pil, face_bbox)

        # ── Step 6: 프롬프트 ─────────────────────────────────────────────────
        prompt, neg_prompt, guidance = self._build_prompt(
            hairstyle_text, normalized_color_text, hair_length
        )
        logger.info(f"[SDPipeline] 프롬프트: {prompt}")
        logger.info(f"[SDPipeline] 네거티브: {neg_prompt}")
        logger.info(f"[SDPipeline] guidance_scale: {guidance}")

        # ── Step 7: SD Inpainting ─────────────────────────────────────────────
        gen_images = self._generate(
            img_512, mask_512, canny_512, face_crop_pil, prompt, neg_prompt, guidance, seeds,
            hair_length=hair_length,
        )

        # ── Step 8: Composite → 원본 해상도 ───────────────────────────────────
        # 하이브리드는 부분 선철거된 베이스 위에 short 생성물을 합성한 뒤,
        # cutoff 아래 잔여 긴머리만 추가 정리한다.
        composite_base_rgb = img_rgb_cleaned
        composite_base_bgr = cv2.cvtColor(composite_base_rgb, cv2.COLOR_RGB2BGR)

        candidates: List[Dict[str, Any]] = []
        for gen_idx, (gen_pil, seed) in enumerate(zip(gen_images, seeds)):
            gen_preview_bgr = cv2.cvtColor(np.array(gen_pil), cv2.COLOR_RGB2BGR)
            composited_bgr = self._composite(
                composite_base_bgr, composite_base_rgb,
                gen_pil, hair_mask_for_sd, scale, pad, (W, H),
                protect_mask=feature_protect_mask,   # 얼굴/귀/눈썹 영역 alpha 침범 방지
                hair_length=hair_length,
            )

            if (
                hair_length in ("short", "medium")
                and cutoff_y_for_post is not None
                and long_hair_mask_for_post is not None
            ):
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    post_debug_enabled = debug_images_common is not None and gen_idx == 0
                    if post_debug_enabled:
                        _store_rgb("pipeline_post_composite_input_rgb", post_rgb)
                        if debug_data_common is not None:
                            debug_data_common["postprocess_sequence"] = [
                                "composite_input",
                                "cutoff_cleanup",
                                "residual_cleanup",
                            ]
                    remnant_info = self._build_real_short_remnant_mask(
                        post_rgb,
                        original_long_hair_mask=long_hair_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        protect_mask=feature_protect_mask,
                        hair_length=hair_length,
                        return_debug=post_debug_enabled,
                    )
                    if post_debug_enabled:
                        removal_mask_for_post, current_short_mask_for_post, remnant_debug = remnant_info
                        _store_mask("pipeline_short_real_mask", current_short_mask_for_post)
                        _store_mask("pipeline_short_removal_mask", removal_mask_for_post)
                        _store_debug_bundle(remnant_debug)
                    else:
                        removal_mask_for_post, current_short_mask_for_post = remnant_info
                    post_cleanup = self._final_cutoff_cleanup(
                        post_rgb,
                        face_bbox=face_bbox,
                        removal_mask=removal_mask_for_post,
                        cutoff_y=cutoff_y_for_post,
                        shoulder_protect=shoulder_protect_for_post,
                        hair_length=hair_length,
                        current_hair_mask=current_short_mask_for_post,
                        return_debug=post_debug_enabled,
                    )
                    if post_debug_enabled:
                        post_rgb, post_cleanup_debug = post_cleanup
                        _store_debug_bundle(post_cleanup_debug)
                        _store_rgb("pipeline_post_after_cutoff_cleanup_rgb", post_rgb)
                    else:
                        post_rgb = post_cleanup
                    residual_cleanup = self._remove_residual_hair_below_cutoff(
                        post_rgb,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        shoulder_protect=shoulder_protect_for_post,
                        hair_length=hair_length,
                        return_debug=post_debug_enabled,
                    )
                    if post_debug_enabled:
                        post_rgb, residual_cleanup_debug = residual_cleanup
                        _store_debug_bundle(residual_cleanup_debug)
                        _store_rgb("pipeline_post_after_residual_cleanup_rgb", post_rgb)
                    else:
                        post_rgb = residual_cleanup
                    composited_bgr = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"[SDPipeline] short/medium 잔여물 cleanup 실패(무시): {e}")

            if not has_color_request:
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    post_rgb = self._preserve_original_hair_tone(
                        source_rgb=img_rgb,
                        target_rgb=post_rgb,
                        face_bbox=face_bbox,
                    )
                    composited_bgr = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"[SDPipeline] 원본 컬러 유지 보정 실패(무시): {e}")

            post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
            final_hair_mask: Optional[np.ndarray] = None
            if has_color_request or hair_length == "short":
                try:
                    final_hair_mask, _, _ = self._segface_hair_mask(post_rgb, face_bbox)
                except Exception as e:
                    logger.warning(f"[SDPipeline] 최종 hair mask 추출 실패(무시): {e}")

            color_distance: Optional[float] = None
            color_score = 0.0
            if has_color_request and target_hair_lab is not None:
                try:
                    color_distance = self._estimate_hair_color_distance(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        target_lab=target_hair_lab,
                        hair_mask=final_hair_mask,
                    )
                    if color_distance is not None:
                        color_score = float(np.clip(1.0 - (color_distance / 80.0), 0.0, 1.0))
                except Exception as e:
                    logger.warning(f"[SDPipeline] 색상 거리 계산 실패(무시): {e}")

            silhouette_score = 0.0
            rank_score = color_score if (has_color_request and target_hair_lab is not None) else 0.0
            silhouette_metrics: Optional[Dict[str, float]] = None
            if hair_length == "short" and cutoff_y_for_post is not None and long_hair_mask_for_post is not None:
                try:
                    silhouette_metrics = self._estimate_short_hair_silhouette_score(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        original_long_hair_mask=long_hair_mask_for_post,
                        protect_mask=feature_protect_mask,
                        hair_mask=final_hair_mask,
                    )
                    silhouette_score = float(silhouette_metrics.get("score", 0.0))
                    if has_color_request and target_hair_lab is not None:
                        rank_score = float(np.clip(silhouette_score * 0.72 + color_score * 0.28, 0.0, 1.0))
                    else:
                        rank_score = silhouette_score
                except Exception as e:
                    logger.warning(f"[SDPipeline] short silhouette 점수 계산 실패(무시): {e}")

            candidates.append({
                "seed": seed,
                "image_bgr": composited_bgr,
                "preview_bgr": gen_preview_bgr,
                "color_distance": color_distance,
                "color_score": color_score,
                "silhouette_score": silhouette_score,
                "rank_score": rank_score,
                "silhouette_metrics": silhouette_metrics,
                "gen_idx": gen_idx,
            })

        if hair_length == "short" and len(candidates) > 1:
            sortable_count = sum(c["silhouette_metrics"] is not None for c in candidates)
            if sortable_count >= 2:
                candidates.sort(
                    key=lambda c: (
                        -float(c["rank_score"]),
                        float((c.get("silhouette_metrics") or {}).get("remnant_ratio", 1e9)),
                        c["color_distance"] is None,
                        c["color_distance"] if c["color_distance"] is not None else 1e9,
                        c["gen_idx"],
                    )
                )
                logger.info("[SDPipeline] short silhouette + color 가중치 기준으로 결과 재정렬 완료")
            elif has_color_request and target_hair_lab is not None:
                logger.info("[SDPipeline] short silhouette 재정렬 스킵 → 컬러 기준으로만 정렬 시도")
                sortable_count = sum(c["color_distance"] is not None for c in candidates)
                if sortable_count >= 2:
                    candidates.sort(
                        key=lambda c: (
                            c["color_distance"] is None,
                            c["color_distance"] if c["color_distance"] is not None else 1e9,
                            c["gen_idx"],
                        )
                    )
            else:
                logger.info("[SDPipeline] short silhouette 재정렬 스킵 (유효 샘플 부족)")
        elif has_color_request and target_hair_lab is not None and len(candidates) > 1:
            sortable_count = sum(c["color_distance"] is not None for c in candidates)
            if sortable_count >= 2:
                candidates.sort(
                    key=lambda c: (
                        c["color_distance"] is None,
                        c["color_distance"] if c["color_distance"] is not None else 1e9,
                        c["gen_idx"],
                    )
                )
                logger.info("[SDPipeline] 컬러 유사도 기준으로 결과 재정렬 완료")
            else:
                logger.info("[SDPipeline] 컬러 유사도 재정렬 스킵 (유효 샘플 부족)")

        if debug_data_common is not None and candidates:
            debug_data_common["candidate_ranking"] = [
                {
                    "seed": int(c["seed"]),
                    "gen_idx": int(c["gen_idx"]),
                    "rank_score": round(float(c.get("rank_score", 0.0)), 4),
                    "silhouette_score": round(float(c.get("silhouette_score", 0.0)), 4),
                    "color_score": round(float(c.get("color_score", 0.0)), 4),
                    "silhouette_metrics": {
                        k: round(float(v), 4) for k, v in (c.get("silhouette_metrics") or {}).items()
                    },
                }
                for c in candidates
            ]

        results: List[SDInpaintResult] = []
        for rank, cand in enumerate(candidates):
            if debug_images_common is not None and rank == 0:
                debug_images_common["sd_generated_rank0_512"] = cand["preview_bgr"]
            results.append(SDInpaintResult(
                image=cand["image_bgr"],
                image_pil=Image.fromarray(cv2.cvtColor(cand["image_bgr"], cv2.COLOR_BGR2RGB)),
                seed=cand["seed"],
                rank=rank,
                mask_used=mask_source,
                clip_score=float(cand["rank_score"]),
                color_score=float(cand["color_score"]),
                silhouette_score=float(cand.get("silhouette_score", 0.0)),
                rank_score=float(cand["rank_score"]),
                mask=hair_mask,
                face_bbox=face_bbox,
                debug_images=debug_images_common if (debug_images_common is not None and rank == 0) else None,
                debug_data=debug_data_common if (debug_data_common is not None and rank == 0) else None,
            ))

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_bisenet(self) -> None:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from models.face_parsing.model import BiSeNet

        seg_path = PROJECT_ROOT / "pretrained_models" / "seg.pth"
        if not seg_path.exists():
            raise FileNotFoundError(f"BiSeNet 가중치 없음: {seg_path}")

        seg = BiSeNet(n_classes=BISENET_CLASSES, output_size=1024, input_size=512)
        seg.load_state_dict(torch.load(str(seg_path), map_location="cpu"), strict=False)
        seg.eval().requires_grad_(False)
        if self.dtype == torch.float16:
            seg.half()
        seg.to(self.device)
        self._bisenet = seg
        logger.info("[SDPipeline] BiSeNet 로드 완료")

    def _load_sam2(self) -> None:
        if not self.config.use_sam2:
            logger.info("[SDPipeline] SAM2 비활성화 (config.use_sam2=False)")
            return

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from utils.sam2_runtime import create_sam2_predictor_factory

        factory = create_sam2_predictor_factory(
            device=str(self.device),
            auto_download=True,
        )
        if factory is None:
            logger.warning(
                "[SDPipeline] SAM2 factory 생성 실패 (checkpoint 없음 or sam2 미설치). "
                "BiSeNet-only로 진행."
            )
        else:
            self._sam2_factory = factory
            logger.info("[SDPipeline] SAM2 factory 등록 완료")

    def _load_mediapipe(self) -> None:
        import mediapipe as mp
        self._mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        logger.info("[SDPipeline] MediaPipe FaceDetection/FaceMesh 로드 완료")

    def _load_sd_pipeline(self) -> None:
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetInpaintPipeline,
        )
        from diffusers.schedulers import DPMSolverMultistepScheduler

        logger.info(f"[SDPipeline] ControlNet 로드: {CONTROLNET_MODEL_ID}")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID, torch_dtype=self.dtype
        )

        logger.info(f"[SDPipeline] SD Inpainting 로드: {SD_INPAINT_MODEL_ID}")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # DPM-Solver++ 스케줄러 (20~30 steps로 고품질)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )

        # IP-Adapter face
        logger.info(f"[SDPipeline] IP-Adapter 로드: {IP_ADAPTER_WEIGHT}")
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO_ID,
            subfolder="models",
            weight_name=IP_ADAPTER_WEIGHT,
        )
        pipe.set_ip_adapter_scale(self.config.ip_adapter_scale)

        # 메모리 최적화 (PyTorch 2.0+ 기본 SDPA 사용)
        # xformers를 강제 활성화하면 일부 Attention Processor(IP-Adapter)에서
        # Tuple shape error 등 충돌이 발생할 수 있으므로 제거합니다.
        # if self.config.enable_xformers:
        #     try:
        #         pipe.enable_xformers_memory_efficient_attention()
        #         logger.info("[SDPipeline] xformers 활성화")
        #     except Exception:
        #         pass

        pipe.to(self.device)
        self._sd_pipe = pipe
        logger.info("[SDPipeline] SD Pipeline 로드 완료")

    def _load_lama(self) -> None:
        """LaMa (Large Mask Inpainting) 모델 로드"""
        if self._lama is not None:
            return
        from simple_lama_inpainting import SimpleLama
        logger.info("[SDPipeline] LaMa 모델 로드 중...")
        self._lama = SimpleLama()
        logger.info("[SDPipeline] LaMa 로드 완료")

    def _ensure_rgb_image_shape(
        self,
        img: np.ndarray,
        expected_hw: Tuple[int, int],
        *,
        context: str,
    ) -> np.ndarray:
        """
        일부 inpainting 백엔드가 stride padding 결과를 원복하지 않고 반환하는 케이스를 방어한다.
        """
        expected_h, expected_w = expected_hw

        if img.ndim == 2:
            img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGBA2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(np.clip(img, 0, 255).astype(np.uint8), 3, axis=2)
        elif img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(f"{context}: unexpected image shape {img.shape}")
        else:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            elif img.shape[2] > 3:
                img = img[..., :3]

        cur_h, cur_w = img.shape[:2]
        if (cur_h, cur_w) == (expected_h, expected_w):
            return img

        delta_h = cur_h - expected_h
        delta_w = cur_w - expected_w
        if delta_h >= 0 and delta_w >= 0 and delta_h <= 8 and delta_w <= 8:
            logger.warning(
                f"[SDPipeline] {context} shape mismatch → crop to original size: "
                f"got={(cur_h, cur_w)}, expected={(expected_h, expected_w)}"
            )
            return img[:expected_h, :expected_w].copy()

        interpolation = cv2.INTER_AREA if (cur_h > expected_h or cur_w > expected_w) else cv2.INTER_LINEAR
        logger.warning(
            f"[SDPipeline] {context} shape mismatch → resize to original size: "
            f"got={(cur_h, cur_w)}, expected={(expected_h, expected_w)}"
        )
        return cv2.resize(img, (expected_w, expected_h), interpolation=interpolation)

    def _lama_inpaint(
        self,
        img_rgb: np.ndarray,
        mask: np.ndarray,
        force_single_pass: bool = False,
    ) -> np.ndarray:
        """
        LaMa로 대규모 영역 inpainting (progressive 방식).

        대규모 마스크(이미지의 20%+ 영역)인 경우, 바깥 테두리부터 안쪽으로
        단계적으로 인페인팅하여 품질을 높임.

        img_rgb: (H, W, 3) uint8 RGB
        mask: (H, W) float32 [0,1] 또는 uint8 [0,255]
        Returns: (H, W, 3) uint8 RGB
        """
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        else:
            mask_u8 = mask.copy()

        total_pixels = int((mask_u8 > 0).sum())
        image_pixels = mask_u8.shape[0] * mask_u8.shape[1]

        # PIL Image 변환 (simple-lama-inpainting 호환성 보장)
        img_pil = Image.fromarray(img_rgb)
        expected_hw = img_rgb.shape[:2]

        # 작은 마스크 또는 hair-removal 강제 단일 패스는 바로 처리
        if force_single_pass or total_pixels < image_pixels * 0.15:
            mask_pil = Image.fromarray(mask_u8)
            result_pil = self._lama(img_pil, mask_pil)
            return self._ensure_rgb_image_shape(
                np.array(result_pil),
                expected_hw,
                context="lama_single_pass",
            )

        # ── Progressive inpainting: 바깥→안쪽 단계적 처리 ──────────────
        # 큰 마스크를 3단계로 나눠서 테두리부터 인페인팅
        logger.info(
            f"[SDPipeline] LaMa progressive 모드: "
            f"total_pixels={total_pixels}, ratio={total_pixels/image_pixels:.1%}"
        )

        current_img = img_rgb.copy()
        remaining_mask = mask_u8.copy()
        n_stages = 3
        # 각 단계에서 사용할 erosion 커널 크기 (점점 안쪽으로)
        erode_sizes = [31, 21, 0]  # 마지막은 나머지 전부

        for stage_i, erode_k in enumerate(erode_sizes):
            if remaining_mask.sum() == 0:
                break

            if erode_k > 0:
                # remaining_mask에서 erosion → 안쪽 영역 제거 → 테두리만 남김
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
                inner = cv2.erode(remaining_mask, k, iterations=1)
                stage_mask = remaining_mask - inner  # 테두리 밴드
                stage_mask = np.clip(stage_mask, 0, 255).astype(np.uint8)
            else:
                # 마지막 단계: 남은 영역 전부
                stage_mask = remaining_mask.copy()

            stage_pixels = int((stage_mask > 0).sum())
            if stage_pixels < 100:
                continue

            logger.info(
                f"[SDPipeline] LaMa stage {stage_i+1}/{n_stages}: "
                f"pixels={stage_pixels}"
            )

            # 이 단계의 마스크로 인페인팅
            img_pil_stage = Image.fromarray(current_img)
            mask_pil_stage = Image.fromarray(stage_mask)
            result_pil = self._lama(img_pil_stage, mask_pil_stage)
            current_img = self._ensure_rgb_image_shape(
                np.array(result_pil),
                expected_hw,
                context=f"lama_progressive_stage_{stage_i + 1}",
            )

            # 처리 완료된 부분 제거
            remaining_mask = np.clip(
                remaining_mask.astype(np.int16) - stage_mask.astype(np.int16),
                0, 255
            ).astype(np.uint8)

        logger.info("[SDPipeline] LaMa progressive 완료")
        return current_img

    def _cv2_inpaint_region(
        self,
        img_rgb: np.ndarray,
        mask: np.ndarray,
        protect_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        cv2 TELEA + NS 블렌딩 기반 부분 인페인팅.
        short/medium 하이브리드 pre-clean의 빠른 기본 경로로 사용한다.
        """
        H, W = img_rgb.shape[:2]
        if mask.shape != (H, W):
            return img_rgb

        mask_u8 = (np.clip(mask, 0.0, 1.0) > 0.35).astype(np.uint8) * 255
        if int((mask_u8 > 0).sum()) < 40:
            return img_rgb

        mask_u8 = cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        if protect_mask is not None and protect_mask.shape == (H, W):
            protect_u8 = (np.clip(protect_mask, 0.0, 1.0) > 0.35).astype(np.uint8) * 255
            protect_u8 = cv2.dilate(
                protect_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(protect_u8))

        if int((mask_u8 > 0).sum()) < 40:
            return img_rgb

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        telea = cv2.inpaint(img_bgr, mask_u8, 5, cv2.INPAINT_TELEA)
        ns = cv2.inpaint(img_bgr, mask_u8, 5, cv2.INPAINT_NS)

        alpha = cv2.GaussianBlur(
            mask_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=5.0,
            sigmaY=5.0,
        )
        alpha = np.clip(alpha, 0.0, 1.0)[..., np.newaxis]
        blend_bgr = telea.astype(np.float32) * 0.68 + ns.astype(np.float32) * 0.32
        out_bgr = blend_bgr * alpha + img_bgr.astype(np.float32) * (1.0 - alpha)
        return cv2.cvtColor(np.clip(out_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    def _load_segface(self) -> None:
        """SegFace (Swin-B) 모델 로드"""
        if self._segface is not None:
            return

        from models.segface.models.segface_celeb import SegFaceCeleb
        from huggingface_hub import hf_hub_download

        logger.info("[SDPipeline] SegFace (Swin-B) 로드 중...")
        segface = SegFaceCeleb(input_resolution=512, model="swin_base")
        
        ckpt_path = hf_hub_download(
            repo_id="kartiknarayan/SegFace", 
            filename="swinb_celeba_512/model_299.pt"
        )
        
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # SegFace의 체크포인트 구조에 맞게 state_dict_backbone만 추출하여 로드
        if "state_dict_backbone" in ckpt:
            segface.load_state_dict(ckpt["state_dict_backbone"], strict=False)
        else:
            segface.load_state_dict(ckpt, strict=False)
        # SegFace는 항상 float32로 실행 (내부에 dtype=torch.float32 하드코딩 있음)
        segface.float().to(self.device).eval()
        self._segface = segface
        logger.info("[SDPipeline] SegFace 로드 완료")

    def _load_segface_hair(self) -> None:
        """Optional custom SegFace hair-only model 로드"""
        if self._segface_hair is not None:
            return

        repo_id = str(getattr(self.config, "segface_hair_repo_id", "") or "").strip()
        if not repo_id:
            logger.info("[SDPipeline] custom hair SegFace 비활성화")
            return

        from huggingface_hub import snapshot_download

        revision = str(getattr(self.config, "segface_hair_revision", "main") or "main").strip()
        token = os.environ.get("HF_TOKEN") or None
        logger.info(f"[SDPipeline] custom hair SegFace 다운로드/로드: repo={repo_id}@{revision}")
        repo_root = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=token,
                allow_patterns=[
                    "best.pt",
                    "config.json",
                    "README.md",
                    "hair_mask_dataset/__init__.py",
                    "hair_mask_dataset/*.py",
                    "models/__init__.py",
                    "models/segface/__init__.py",
                    "models/segface/models/__init__.py",
                    "models/segface/models/*.py",
                    "__init__.py",
                ],
            )
        )

        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        segface_hair_module = importlib.import_module("hair_mask_dataset.segface_hair_model")
        SegFaceHairModel = getattr(segface_hair_module, "SegFaceHairModel")

        ckpt_path = repo_root / "best.pt"
        cfg_path = repo_root / "config.json"
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config")
        if cfg is None:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        segface_hair = SegFaceHairModel(
            input_resolution=int(cfg.get("image_size", 512)),
            model_name=str(cfg.get("model_name", "swin_base")),
            load_pretrained=False,
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
            lora_rank=int(cfg.get("lora_rank", 0)),
            lora_alpha=float(cfg.get("lora_alpha", 16.0)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            lora_targets=tuple(cfg.get("lora_targets", ())),
        )
        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError(f"Unexpected custom hair checkpoint format: keys={list(checkpoint.keys())[:20]}")
        segface_hair.load_state_dict(model_state, strict=False)
        segface_hair.float().to(self.device).eval()

        self._segface_hair = segface_hair
        self._segface_hair_threshold = float(cfg.get("threshold", 0.5))
        logger.info(
            "[SDPipeline] custom hair SegFace 로드 완료 "
            f"(threshold={self._segface_hair_threshold:.2f})"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Segmentation: SegFace + SAM2
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_face(
        self, img_rgb: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """MediaPipe로 얼굴 bbox (x1, y1, x2, y2) 반환"""
        H, W = img_rgb.shape[:2]
        result = self._mp_face.process(img_rgb)
        if not result.detections:
            return None
        bb = result.detections[0].location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * W))
        y1 = max(0, int(bb.ymin * H))
        x2 = min(W, int((bb.xmin + bb.width) * W))
        y2 = min(H, int((bb.ymin + bb.height) * H))
        return (x1, y1, x2, y2)

    def _detect_face_mesh(
        self, img_rgb: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        MediaPipe FaceMesh 랜드마크 검출.

        Returns:
            landmarks_norm: (N, 3) float32, normalized [0,1] 좌표
            landmarks_px:   (N, 2) int32, 원본 픽셀 좌표
        """
        H, W = img_rgb.shape[:2]
        if self._mp_face_mesh is None:
            return None, None

        result = self._mp_face_mesh.process(img_rgb)
        if not result.multi_face_landmarks:
            return None, None

        lms = result.multi_face_landmarks[0].landmark
        if not lms:
            return None, None

        landmarks_norm = np.asarray([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        xs = np.clip(np.round(landmarks_norm[:, 0] * W), 0, W - 1).astype(np.int32)
        ys = np.clip(np.round(landmarks_norm[:, 1] * H), 0, H - 1).astype(np.int32)
        landmarks_px = np.stack([xs, ys], axis=1)
        return landmarks_norm, landmarks_px

    @staticmethod
    def _build_face_mesh_analysis(
        landmarks_norm: np.ndarray,
        landmarks_px: np.ndarray,
    ) -> Dict[str, Any]:
        """
        얼굴형 분석용 FaceMesh 메타데이터 생성.
        """
        n = int(landmarks_norm.shape[0])

        def _safe_dist(i: int, j: int) -> Optional[float]:
            if i >= n or j >= n:
                return None
            p = landmarks_px[i].astype(np.float32)
            q = landmarks_px[j].astype(np.float32)
            return float(np.linalg.norm(p - q))

        face_height = _safe_dist(10, 152)    # forehead(top) ~ chin
        cheekbone_width = _safe_dist(234, 454)
        jaw_width = _safe_dist(172, 397)
        temple_width = _safe_dist(127, 356)

        ratios: Dict[str, Optional[float]] = {
            "cheekbone_to_height": None,
            "jaw_to_height": None,
            "temple_to_height": None,
            "jaw_to_cheekbone": None,
        }
        if face_height and face_height > 1e-6:
            if cheekbone_width is not None:
                ratios["cheekbone_to_height"] = cheekbone_width / face_height
            if jaw_width is not None:
                ratios["jaw_to_height"] = jaw_width / face_height
            if temple_width is not None:
                ratios["temple_to_height"] = temple_width / face_height
        if cheekbone_width and cheekbone_width > 1e-6 and jaw_width is not None:
            ratios["jaw_to_cheekbone"] = jaw_width / cheekbone_width

        keypoints: Dict[str, Any] = {}
        keypoint_map = {
            "forehead_top": 10,
            "chin": 152,
            "left_cheekbone": 234,
            "right_cheekbone": 454,
            "left_jaw": 172,
            "right_jaw": 397,
            "left_temple": 127,
            "right_temple": 356,
        }
        for name, idx in keypoint_map.items():
            if idx < n:
                keypoints[name] = {
                    "index": idx,
                    "norm": [
                        float(landmarks_norm[idx, 0]),
                        float(landmarks_norm[idx, 1]),
                        float(landmarks_norm[idx, 2]),
                    ],
                    "px": [int(landmarks_px[idx, 0]), int(landmarks_px[idx, 1])],
                }

        return {
            "landmarks_count": n,
            "landmarks_norm": landmarks_norm.astype(float).round(6).tolist(),
            "landmarks_px": landmarks_px.astype(int).tolist(),
            "metrics_px": {
                "face_height": face_height,
                "cheekbone_width": cheekbone_width,
                "jaw_width": jaw_width,
                "temple_width": temple_width,
            },
            "ratios": ratios,
            "keypoints": keypoints,
        }

    def _render_face_mesh_debug_images(
        self,
        img_rgb: np.ndarray,
        landmarks_px: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        FaceMesh 디버그 이미지 생성 (BGR).
        """
        import mediapipe as mp

        H, W = img_rgb.shape[:2]
        base_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        points_bgr = base_bgr.copy()
        tess_bgr = base_bgr.copy()
        contour_bgr = base_bgr.copy()

        # points
        for x, y in landmarks_px:
            cv2.circle(points_bgr, (int(x), int(y)), 1, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

        # tessellation
        for a, b in mp.solutions.face_mesh.FACEMESH_TESSELATION:
            if a >= len(landmarks_px) or b >= len(landmarks_px):
                continue
            p1 = tuple(int(v) for v in landmarks_px[a])
            p2 = tuple(int(v) for v in landmarks_px[b])
            cv2.line(tess_bgr, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)

        # contours
        for a, b in mp.solutions.face_mesh.FACEMESH_CONTOURS:
            if a >= len(landmarks_px) or b >= len(landmarks_px):
                continue
            p1 = tuple(int(v) for v in landmarks_px[a])
            p2 = tuple(int(v) for v in landmarks_px[b])
            cv2.line(contour_bgr, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)

        # face oval mask
        oval_idxs = sorted(
            {i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge}
        )
        oval_mask = np.zeros((H, W), dtype=np.uint8)
        if oval_idxs:
            pts = np.asarray([landmarks_px[i] for i in oval_idxs if i < len(landmarks_px)], dtype=np.int32)
            if len(pts) >= 3:
                hull = cv2.convexHull(pts.reshape(-1, 1, 2))
                cv2.fillConvexPoly(oval_mask, hull, 255)
        oval_mask_bgr = cv2.cvtColor(oval_mask, cv2.COLOR_GRAY2BGR)

        return {
            "mediapipe_face_mesh_points": points_bgr,
            "mediapipe_face_mesh_tessellation": tess_bgr,
            "mediapipe_face_mesh_contours": contour_bgr,
            "mediapipe_face_mesh_oval_mask": oval_mask_bgr,
        }

    def _build_feature_protect_mask(
        self,
        face_region_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmarks_px: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        얼굴 보호 마스크를 보강한다.
        - SegFace face_region_mask를 기본으로 사용
        - 눈썹/눈 주변은 FaceMesh로 정밀 보호
        - 귀/옆얼굴은 얇은 side pad를 추가해 가짜 귀 생성 방지
        """
        H, W = face_region_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)

        base_u8 = (np.clip(face_region_mask, 0.0, 1.0) > 0.35).astype(np.uint8) * 255
        protect_u8 = cv2.dilate(
            base_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    max(9, int(face_w * 0.10)) | 1,
                    max(9, int(face_h * 0.08)) | 1,
                ),
            ),
            iterations=1,
        )

        # eyebrow/eye band 보호: bangs가 눈썹을 재합성하지 않게 막는다.
        if landmarks_px is not None and len(landmarks_px) > 0:
            try:
                import mediapipe as mp

                eye_brow_idxs = sorted({
                    i
                    for edges in (
                        mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                        mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                        mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
                        mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
                    )
                    for edge in edges
                    for i in edge
                    if i < len(landmarks_px)
                })
                if eye_brow_idxs:
                    pts = np.asarray([landmarks_px[i] for i in eye_brow_idxs], dtype=np.int32)
                    if len(pts) >= 3:
                        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
                        eye_brow_mask = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillConvexPoly(eye_brow_mask, hull, 255)
                        eye_brow_mask = cv2.dilate(
                            eye_brow_mask,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (
                                    max(19, int(face_w * 0.18)) | 1,
                                    max(13, int(face_h * 0.12)) | 1,
                                ),
                            ),
                            iterations=1,
                        )
                        protect_u8 = cv2.bitwise_or(protect_u8, eye_brow_mask)
            except Exception:
                pass
        else:
            brow_band = np.zeros((H, W), dtype=np.uint8)
            brow_band[
                max(0, int(y1 + face_h * 0.14)):min(H, int(y1 + face_h * 0.46)),
                max(0, int(x1 - face_w * 0.06)):min(W, int(x2 + face_w * 0.06)),
            ] = 255
            protect_u8 = cv2.bitwise_or(protect_u8, brow_band)

        # 귀/옆얼굴 보호: SD가 가짜 귀를 새로 그리는 것을 막는다.
        ear_mask = np.zeros((H, W), dtype=np.uint8)
        ear_y = int(y1 + face_h * 0.52)
        ear_axes = (
            max(10, int(face_w * 0.11)),
            max(18, int(face_h * 0.20)),
        )
        left_center = (max(0, int(x1 - face_w * 0.02)), ear_y)
        right_center = (min(W - 1, int(x2 + face_w * 0.02)), ear_y)
        cv2.ellipse(ear_mask, left_center, ear_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(ear_mask, right_center, ear_axes, 0, 0, 360, 255, -1)
        ear_mask[:max(0, int(y1 + face_h * 0.10)), :] = 0
        ear_mask[min(H, int(y2 - face_h * 0.06)):, :] = 0
        protect_u8 = cv2.bitwise_or(protect_u8, ear_mask)

        # forehead 전체를 막지는 않되, brow band 위 경계는 부드럽게 보호
        protect_u8 = cv2.GaussianBlur(protect_u8.astype(np.float32) / 255.0, (0, 0), 2.0, 2.0)
        return np.clip(protect_u8, 0.0, 1.0).astype(np.float32)

    def _segface_hair_mask(self, img_rgb: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SegFace로 머리카락 + 얼굴 + 옷 영역 마스크 생성.
        얼굴 bbox를 기준으로 여유 있게 크롭한 뒤 512x512로 리사이즈하여 SegFace에 입력.
        이후 원본 해상도의 전체 영역으로 다시 복원하여 출력.

        Returns:
            hair_mask  (H×W float32): 머리카락 영역
            face_mask  (H×W float32): 얼굴/목/귀 영역 (inpaint에서 보호)
            cloth_mask (H×W float32): 옷 영역 (inpaint에서 보호)
        """
        H, W = img_rgb.shape[:2]
        x1, y1, x2, y2 = face_bbox
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw // 2, y1 + bh // 2
        
        # CelebA-HQ 스타일 크롭: 얼굴 bbox 기준 박스 크기를 약 2.5~3배로 키워서 머리카락 전체를 포함
        box_size = int(max(bw, bh) * 2.8)
        # 윗머리가 잘리지 않도록 크롭 중심을 얼굴보다 조금 위로 (10%) 올림
        cy = max(0, cy - int(box_size * 0.1))
        
        crop_x1 = max(0, cx - box_size // 2)
        crop_y1 = max(0, cy - box_size // 2)
        crop_x2 = min(W, crop_x1 + box_size)
        crop_y2 = min(H, crop_y1 + box_size)
        
        cw = crop_x2 - crop_x1
        ch = crop_y2 - crop_y1
        
        # 정사각형 형태로 패딩해서 512x512 로 만들기 위한 준비
        crop_max = max(cw, ch)
        pad_bottom = crop_max - ch
        pad_right = crop_max - cw
        
        # 크롭
        crop_img = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        # 패딩 (검은 배경)
        if pad_bottom > 0 or pad_right > 0:
            crop_img = cv2.copyMakeBorder(crop_img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
            
        crop_h, crop_w = crop_img.shape[:2]
        
        # 512x512 변환
        inp_np = cv2.resize(crop_img, (512, 512), interpolation=cv2.INTER_AREA)
        
        # SegFace 입력 형식: [0, 1]로 Normalize (ImageNet mean/std 사용)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp_t = (inp_np / 255.0 - mean) / std
        inp_t = torch.from_numpy(inp_t).float().permute(2, 0, 1).unsqueeze(0)
        
        # SegFace는 float32로 고정 실행 (모델 내부 float32 하드코딩 때문에 half 금지)
        inp_t = inp_t.float().to(self.device)

        with torch.no_grad():
            DUMMY_LABELS = None
            DUMMY_DATASET = None
            logits = self._segface(inp_t, DUMMY_LABELS, DUMMY_DATASET)
            # logits: [1, 19, 512, 512]
            parsing = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        hair_512: np.ndarray
        if self._segface_hair is not None:
            with torch.no_grad():
                hair_out = self._segface_hair(inp_t)
                hair_logits = hair_out["hair_logits"]
                hair_probs = torch.sigmoid(hair_logits).squeeze(0).squeeze(0).cpu().numpy()
            hair_512 = (hair_probs >= float(self._segface_hair_threshold)).astype(np.float32)
        else:
            hair_512 = (parsing == HAIR_CLASS_IDX).astype(np.float32)
        face_512  = np.isin(parsing, list(FACE_CLASS_IDXS)).astype(np.float32)
        cloth_512 = (parsing == CLOTH_CLASS_IDX).astype(np.float32)

        # 1. 원본 비율 (crop_w, crop_h) 해상도로 다시 리사이즈
        hair_crop  = cv2.resize(hair_512,  (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        face_crop  = cv2.resize(face_512,  (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        cloth_crop = cv2.resize(cloth_512, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        # 2. 패딩 부분 잘라내기
        hair_crop  = hair_crop[:ch, :cw]
        face_crop  = face_crop[:ch, :cw]
        cloth_crop = cloth_crop[:ch, :cw]

        # 3. 원본 HxW 해상도에 덮어쓰기
        hair_orig  = np.zeros((H, W), dtype=np.float32)
        face_orig  = np.zeros((H, W), dtype=np.float32)
        cloth_orig = np.zeros((H, W), dtype=np.float32)

        hair_orig[crop_y1:crop_y2, crop_x1:crop_x2]  = hair_crop
        face_orig[crop_y1:crop_y2, crop_x1:crop_x2]  = face_crop
        cloth_orig[crop_y1:crop_y2, crop_x1:crop_x2] = cloth_crop

        return (
            (hair_orig  > 0.5).astype(np.float32),
            (face_orig  > 0.5).astype(np.float32),
            (cloth_orig > 0.5).astype(np.float32),
        )

    def _refine_with_sam2(
        self,
        img_rgb: np.ndarray,           # H×W×3 RGB
        base_mask: np.ndarray,          # H×W float32
        face_bbox: Tuple[int, int, int, int],
        prompt_text: str,
    ) -> Tuple[np.ndarray, str]:
        """
        SAM2로 SegFace 마스크를 정밀 보정.

        Returns:
            (refined_mask H×W float32, source_name)
        """
        if self._sam2_factory is None:
            # SAM2 없으면 BiSeNet mask에만 dilate 적용
            return self._dilate_mask(base_mask), "bisenet"

        try:
            predictor = self._sam2_factory()
            H, W = img_rgb.shape[:2]
            x1, y1, x2, y2 = face_bbox

            bw = x2 - x1
            bh = y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # SAM2 bbox: 긴 머리 고려해서 하단을 BiSeNet hair 최하단까지 확장
            hair_coords = np.argwhere(base_mask > 0.5)  # (N,2) [row, col]
            if len(hair_coords) > 0:
                hair_bottom = int(hair_coords[:, 0].max())
                bbox_bottom = min(H - 1, max(hair_bottom + 20, y2 + int(bh * 0.2)))
            else:
                bbox_bottom = min(H - 1, y2 + int(bh * 0.6))  # fallback: 얼굴 높이 60% 아래

            sam_bbox = np.array([
                max(0,     x1 - int(bw * 0.6)),
                max(0,     y1 - int(bh * 0.6)),
                min(W - 1, x2 + int(bw * 0.6)),
                bbox_bottom,
            ], dtype=np.float32)

            # Positive points: 정수리/옆머리 + 긴 머리 흘러내리는 옆쪽
            hair_top_y  = max(5, y1 - int(bh * 0.25))   # 정수리
            side_y      = max(5, y1 - int(bh * 0.05))   # 귀 위쪽
            long_hair_y = min(H - 5, y2 + int(bh * 0.4)) # 턱 아래 긴 머리
            pos_pts = np.array([
                [cx,                    hair_top_y],   # 정수리 중앙
                [cx - int(bw * 0.25),   hair_top_y],   # 정수리 왼쪽
                [cx + int(bw * 0.25),   hair_top_y],   # 정수리 오른쪽
                [x1 - int(bw * 0.05),   side_y],       # 왼쪽 옆머리
                [x2 + int(bw * 0.05),   side_y],       # 오른쪽 옆머리
                [x1 - int(bw * 0.2),    long_hair_y],  # 왼쪽 긴 머리
                [x2 + int(bw * 0.2),    long_hair_y],  # 오른쪽 긴 머리
            ], dtype=np.float32)
            pos_pts[:, 0] = np.clip(pos_pts[:, 0], 0, W - 1)
            pos_pts[:, 1] = np.clip(pos_pts[:, 1], 0, H - 1)

            # Negative points: 얼굴 격자 9점 + 목/상체 중앙 (몸통 잡지 않도록)
            neck_y   = min(H - 5, y2 + int(bh * 0.15))
            body_y   = min(H - 5, y2 + int(bh * 0.5))
            neg_pts = np.array([
                # 얼굴 상단부 (이마)
                [x1 + int(bw * 0.25), y1 + int(bh * 0.25)],
                [cx,                   y1 + int(bh * 0.25)],
                [x2 - int(bw * 0.25), y1 + int(bh * 0.25)],
                # 얼굴 중앙부 (눈/코)
                [x1 + int(bw * 0.25), cy],
                [cx,                   cy],
                [x2 - int(bw * 0.25), cy],
                # 얼굴 하단부 (입/턱)
                [x1 + int(bw * 0.25), y2 - int(bh * 0.15)],
                [cx,                   y2 - int(bh * 0.15)],
                [x2 - int(bw * 0.25), y2 - int(bh * 0.15)],
                # 목/상체 중앙 (긴 머리가 옆으로 흘러도 몸통 중앙은 제외)
                [cx,  neck_y],
                [cx,  body_y],
            ], dtype=np.float32)
            # 이미지 범위 클램프
            neg_pts[:, 0] = np.clip(neg_pts[:, 0], 0, W - 1)
            neg_pts[:, 1] = np.clip(neg_pts[:, 1], 0, H - 1)

            point_coords = np.concatenate([pos_pts, neg_pts], axis=0)
            point_labels = np.concatenate([
                np.ones(len(pos_pts),  dtype=np.int32),
                np.zeros(len(neg_pts), dtype=np.int32),
            ])

            # SAM2 predict (multimask=True → 가장 face overlap 적은 마스크 선택)
            predictor.set_image(img_rgb)
            prediction = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=sam_bbox[None, :],
                multimask_output=True,
            )

            # predict() 반환 형태: dict | (masks, scores, logits) tuple
            if isinstance(prediction, dict):
                masks = prediction.get("masks")
            elif isinstance(prediction, (tuple, list)):
                # (masks, iou_scores, low_res_logits) 형태로 반환
                masks = prediction[0]
                # 드물게 masks 자체가 또 tuple/list인 경우 unwrap
                while isinstance(masks, (tuple, list)):
                    masks = masks[0]
            else:
                masks = prediction

            if masks is not None:
                # numpy/tensor → numpy 변환
                if hasattr(masks, "cpu"):
                    masks_np = masks.cpu().numpy()
                else:
                    masks_np = np.asarray(masks)

                # shape 정규화: (N,H,W) or (H,W)
                if masks_np.ndim == 2:
                    masks_np = masks_np[np.newaxis]  # → (1,H,W)
                elif masks_np.ndim != 3 or masks_np.shape[0] == 0:
                    raise ValueError(f"Unexpected SAM2 mask shape: {masks_np.shape}")

                # multimask: SegFace의 base_mask와 가장 일치하는(IoU가 높은) 마스크를 선택
                best_mask = None
                best_iou = -1.0
                
                # base_mask (SegFace 예측 결과)
                base_f = (base_mask > 0.5).astype(np.float32)
                base_sum = base_f.sum()
                
                for m in masks_np:
                    m_f = (m > 0.5).astype(np.float32)
                    if m_f.shape != (H, W):
                        m_f = cv2.resize(m_f, (W, H), interpolation=cv2.INTER_LINEAR)
                        m_f = (m_f > 0.5).astype(np.float32)
                    
                    # Compute IoU with base_mask
                    intersection = (m_f * base_f).sum()
                    union = m_f.sum() + base_sum - intersection
                    iou = intersection / (union + 1e-6)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = m_f

                refined_np = best_mask
                
                # 얼굴/몸통 등 잘못된 영역이 넓게 잡히는 것을 방지하기 위해 
                # SegFace base_mask_dilated 와의 교집합만 취함
                base_mask_dilated = self._dilate_mask(base_mask)
                refined_np = np.clip(refined_np * base_mask_dilated, 0.0, 1.0)
                
                if refined_np.sum() < 300:
                    logger.warning("[SDPipeline] SAM2 결과가 너무 작아 SegFace로 폴백")
                    return self._dilate_mask(base_mask), "segface"

                return self._dilate_mask(refined_np), "sam2"

        except Exception as e:
            logger.warning(f"[SDPipeline] SAM2 실패, BiSeNet으로 폴백: {e}")

        return self._dilate_mask(base_mask), "bisenet"

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """마스크 dilate (경계 확장)"""
        px = self.config.mask_dilate_px
        if px <= 0:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return np.clip(dilated, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _sample_mask_points(
        hair_coords: np.ndarray, n: int = 3
    ) -> np.ndarray:
        """hair mask 좌표에서 대표 n개 point 샘플링 (row, col → x, y)"""
        if len(hair_coords) == 0:
            return np.empty((0, 2), dtype=np.float32)
        idx = np.linspace(0, len(hair_coords) - 1, n, dtype=int)
        pts = hair_coords[idx]  # (n, 2) [row, col]
        return pts[:, ::-1].astype(np.float32)  # → (n, 2) [x=col, y=row]

    # ──────────────────────────────────────────────────────────────────────────
    # SD Input Preparation
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_sd_inputs(
        self,
        img_rgb: np.ndarray,     # H×W×3 RGB
        hair_mask: np.ndarray,   # H×W float32
        mask_edge_suppression: float = 1.0,  # 0.0=엣지 보존, 1.0=마스크 내부 엣지 완전 제거
        canny_suppress_mask: Optional[np.ndarray] = None,  # H×W float32 — 이 영역의 canny edge도 제거
    ) -> Tuple[Image.Image, Image.Image, Image.Image, float, Tuple[int, int]]:
        """
        Letter-box resize → 512×512.

        Args:
            canny_suppress_mask: short/medium에서 사용. 기존 long-hair 영역의 canny edge를
                                 추가로 제거하여 ControlNet이 원본 긴머리 윤곽을 따라가지 않게 함.

        Returns:
            img_512:    PIL RGB 512×512 (full image)
            mask_512:   PIL L  512×512 (흰색=inpaint)
            canny_512:  PIL RGB 512×512 (ControlNet conditioning)
            scale:      resize 비율
            pad:        (pad_left, pad_top) pixels
        """
        H, W = img_rgb.shape[:2]
        scale = SD_SIZE / max(H, W)
        new_w, new_h = int(W * scale), int(H * scale)
        pad_l = (SD_SIZE - new_w) // 2
        pad_t = (SD_SIZE - new_h) // 2

        # ── image letterbox
        img_rs = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((SD_SIZE, SD_SIZE, 3), dtype=np.uint8)
        canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = img_rs

        # ── mask letterbox
        msk_rs = cv2.resize(hair_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
        msk_canvas = np.zeros((SD_SIZE, SD_SIZE), dtype=np.float32)
        msk_canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = msk_rs

        # ── Canny edge
        # 기본(헤어 생성): 마스크 내부 엣지 강하게 제거
        # 배경 복원(fill): 일부 엣지를 남겨 texture/구조 연속성 확보
        gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
        suppress = float(np.clip(mask_edge_suppression, 0.0, 1.0))
        hair_hard = (msk_canvas > 0.5).astype(np.float32)

        # canny_suppress_mask가 있으면 해당 영역의 edge도 완전 제거
        # → LaMa 잔여 블러 윤곽이 ControlNet에 전달되지 않음
        if canny_suppress_mask is not None:
            sup_rs = cv2.resize(canny_suppress_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sup_canvas = np.zeros((SD_SIZE, SD_SIZE), dtype=np.float32)
            sup_canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = sup_rs
            # dilate: 경계 blur 잔여물까지 제거
            k_sup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            sup_hard = cv2.dilate(
                (sup_canvas > 0.3).astype(np.uint8), k_sup, iterations=1
            ).astype(np.float32)
            # 기존 hair_hard와 합쳐서 최종 suppression 영역
            hair_hard = np.clip(hair_hard + sup_hard, 0.0, 1.0)
            logger.info(
                f"[SDPipeline] canny suppress 확장: "
                f"gen_mask pixels={int((msk_canvas > 0.5).sum())}, "
                f"total suppress pixels={int((hair_hard > 0.5).sum())}"
            )

        canny_f = canny.astype(np.float32) * (1.0 - hair_hard * suppress)
        canny_rgb = cv2.cvtColor(canny_f.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        img_512   = Image.fromarray(canvas)
        mask_512  = Image.fromarray((msk_canvas * 255).astype(np.uint8), mode="L")
        canny_512 = Image.fromarray(canny_rgb)

        return img_512, mask_512, canny_512, scale, (pad_l, pad_t)

    def _crop_face(
        self,
        img_pil: Image.Image,
        face_bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        """IP-Adapter용 얼굴 crop (얼굴만 — 머리카락 최소화)

        padding을 아래쪽은 넉넉히(턱/목 포함), 위쪽/옆은 최소화(머리카락 제외)
        IP-Adapter가 원본 헤어 스타일을 conditioning하면 숏컷 변환이 안 됨.
        """
        x1, y1, x2, y2 = face_bbox
        W, H = img_pil.size
        bw, bh = x2 - x1, y2 - y1
        # 위/옆은 패딩 최소화(0.05) → 머리카락 포함 억제
        # 아래는 패딩 넉넉히(0.2) → 턱/목 포함 → 얼굴 identity 안정화
        pad_side = int(bw * 0.05)
        pad_top  = int(bh * 0.05)
        pad_bot  = int(bh * 0.20)
        crop = img_pil.crop((
            max(0, x1 - pad_side),
            max(0, y1 - pad_top),
            min(W, x2 + pad_side),
            min(H, y2 + pad_bot),
        ))
        return crop.resize((224, 224), Image.LANCZOS)

    # ──────────────────────────────────────────────────────────────────────────
    # Hair Length Classification
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_hair_length(hairstyle_text: str) -> str:
        """헤어스타일 텍스트 → 'short' | 'medium' | 'long'"""
        text = hairstyle_text.lower()
        for kw in _SHORT_HAIR_KEYWORDS:
            if kw in text:
                return "short"
        for kw in _MEDIUM_HAIR_KEYWORDS:
            if kw in text:
                return "medium"
        return "long"

    def _expand_mask_for_short_hair(
        self,
        hair_mask: np.ndarray,              # H×W float32
        face_bbox: Tuple[int, int, int, int],
        H: int,
        W: int,
        hair_length: str = "short",
    ) -> np.ndarray:
        """
        단발/중단발 변환 시 마스크 하단 확장.

        긴 머리 → 단발로 바꿀 때, 현재 긴 머리 마스크 하단(턱 아래)에도 마스크를
        씌워 SD가 그 영역을 배경/피부로 채우도록 유도함.
        확장 없이 그냥 두면 원본 긴 머리 픽셀이 composite에서 살아남음.
        """
        x1, y1, x2, y2 = face_bbox
        face_h = max(y2 - y1, 1)

        # 길이별 기준점: 어디부터 "머리카락이 없어야 하는가"
        if hair_length == "short":
            # 턱~윗목 사이 (얼굴 높이의 +12%)
            cutoff_y = int(y2 + face_h * 0.12)
        else:  # medium
            # 어깨 위 (얼굴 높이의 +60%)
            cutoff_y = int(y2 + face_h * 0.60)
        cutoff_y = max(0, min(cutoff_y, H - 1))

        # 마스크 확장 한계: 어깨 아래(얼굴 높이 2배)를 넘지 않도록 제한 (옷 보호)
        max_expand_y = int(y2 + face_h * 2.0)
        max_expand_y = min(max_expand_y, H - 1)

        # SAM2가 한쪽만 끊기는 문제 보정: 좌/우 각각 최하단을 구해서 더 긴 쪽에 맞춤
        cx = (x1 + x2) // 2
        left_rows  = np.any(hair_mask[:, :cx] > 0.5, axis=1)
        right_rows = np.any(hair_mask[:, cx:] > 0.5, axis=1)

        left_bottom  = int(np.max(np.where(left_rows)))  if np.any(left_rows)  else cutoff_y
        right_bottom = int(np.max(np.where(right_rows))) if np.any(right_rows) else cutoff_y

        # 좌우 중 더 긴 쪽을 기준으로 반대쪽도 같은 높이까지 확장 (대칭 보정)
        lowest_hair_y = min(max(left_bottom, right_bottom), max_expand_y)

        if lowest_hair_y <= cutoff_y:
            return hair_mask

        # cutoff_y ~ lowest_hair_y 구간을 마스크에 추가
        # 헤어가 실제로 있는 열(column) 범위를 위쪽 행들에서 추정
        hair_cols_all = np.where(np.any(hair_mask[:cutoff_y] > 0.5, axis=0))[0]
        if len(hair_cols_all) > 0:
            default_c_min = int(hair_cols_all.min())
            default_c_max = int(hair_cols_all.max())
        else:
            default_c_min, default_c_max = x1, x2

        expanded = hair_mask.copy()
        for row in range(cutoff_y, min(lowest_hair_y + 1, H)):
            row_hair_cols = np.where(hair_mask[row] > 0.3)[0]
            if len(row_hair_cols) > 0:
                c_min, c_max = int(row_hair_cols.min()), int(row_hair_cols.max())
            else:
                c_min, c_max = default_c_min, default_c_max
            expanded[row, max(0, c_min):min(W, c_max + 1)] = 1.0

        return expanded

    # ──────────────────────────────────────────────────────────────────────────
    # Color Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_color_text(color_text: str) -> str:
        text = str(color_text or "").strip()
        lowered = text.lower()
        if lowered in _NO_COLOR_HINTS:
            return ""
        return text

    @staticmethod
    def _resolve_target_hair_lab(color_text: str) -> Optional[np.ndarray]:
        query = str(color_text or "").strip().lower()
        if not query:
            return None
        for keyword, rgb in _HAIR_COLOR_TARGET_RGB:
            if keyword in query:
                rgb_np = np.array([[list(rgb)]], dtype=np.uint8)
                lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
                return lab
        return None

    def _estimate_hair_color_distance(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        target_lab: np.ndarray,
        hair_mask: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        if hair_mask is None:
            hair_mask, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
        hair_u8 = (hair_mask > 0.45).astype(np.uint8) * 255
        if int((hair_u8 > 0).sum()) < 80:
            return None

        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hair_pixels = lab[hair_u8 > 0]
        if hair_pixels.shape[0] < 50:
            return None

        # 극단적인 shadow 영역 영향 완화
        if hair_pixels.shape[0] > 200:
            l_vals = hair_pixels[:, 0]
            keep = l_vals > np.percentile(l_vals, 15.0)
            if np.any(keep):
                hair_pixels = hair_pixels[keep]

        med = np.median(hair_pixels, axis=0)
        d_l = abs(float(med[0] - target_lab[0]))
        d_a = abs(float(med[1] - target_lab[1]))
        d_b = abs(float(med[2] - target_lab[2]))
        # 색조(a,b)를 더 강하게 반영
        return 0.25 * d_l + 0.85 * d_a + 0.85 * d_b

    def _estimate_short_hair_silhouette_score(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        original_long_hair_mask: np.ndarray,
        protect_mask: Optional[np.ndarray] = None,
        hair_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        short 후보의 silhouette 품질을 정량화한다.
        남은 long-hair remnant, 목 노출, 턱선 아래 길이, 좌우 side presence를 함께 본다.
        """
        H, W = img_rgb.shape[:2]
        if original_long_hair_mask.shape != (H, W):
            return {
                "score": 0.0,
                "remnant_score": 0.0,
                "neck_clear_score": 0.0,
                "bottom_score": 0.0,
                "presence_score": 0.0,
                "balance_score": 0.0,
                "area_score": 0.0,
            }

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        face_area = float(face_w * face_h)
        cx = int(0.5 * (x1 + x2))

        if hair_mask is None:
            hair_mask, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
        hair_mask = np.clip(hair_mask.astype(np.float32), 0.0, 1.0)
        if protect_mask is not None and protect_mask.shape == (H, W):
            hair_mask = np.clip(hair_mask - np.clip(protect_mask, 0.0, 1.0), 0.0, 1.0)
        hair_u8 = (hair_mask > 0.45).astype(np.uint8) * 255

        remnant_mask, short_real_mask = self._build_real_short_remnant_mask(
            img_rgb,
            original_long_hair_mask=original_long_hair_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            protect_mask=protect_mask,
            hair_length="short",
            current_hair_mask=hair_mask,
        )
        remnant_u8 = (remnant_mask > 0.35).astype(np.uint8) * 255
        short_real_u8 = (short_real_mask > 0.35).astype(np.uint8) * 255

        long_px = int((np.clip(original_long_hair_mask, 0.0, 1.0) > 0.35).sum())
        remnant_px = int((remnant_u8 > 0).sum())
        short_real_px = int((short_real_u8 > 0).sum())

        remnant_ratio = float(remnant_px / max(long_px, 1))
        remnant_score = float(np.clip(1.0 - (remnant_ratio / 0.30), 0.0, 1.0))

        neck_top = int(np.clip(cutoff_y, 0, H))
        neck_bottom = int(np.clip(cutoff_y + face_h * 0.46, 0, H))
        neck_x1 = max(0, int(cx - face_w * 0.18))
        neck_x2 = min(W, int(cx + face_w * 0.18))
        neck_occupancy = 0.0
        if neck_top < neck_bottom and neck_x1 < neck_x2:
            neck_region = hair_u8[neck_top:neck_bottom, neck_x1:neck_x2] > 0
            if neck_region.size > 0:
                neck_occupancy = float(neck_region.mean())
        neck_clear_score = float(np.clip(1.0 - (neck_occupancy / 0.12), 0.0, 1.0))

        search_top = max(0, int(y1 - face_h * 0.08))
        hard_bottom = min(H - 1, int(cutoff_y + face_h * 0.52))
        allowed_bottom = min(H - 1, int(cutoff_y + face_h * 0.16))

        def _lowest_hair_y(xa: int, xb: int) -> int:
            xa = max(0, min(xa, W))
            xb = max(0, min(xb, W))
            if xa >= xb or search_top > hard_bottom:
                return cutoff_y - 1
            region = hair_u8[search_top:hard_bottom + 1, xa:xb] > 0
            rows = np.flatnonzero(np.any(region, axis=1))
            if rows.size == 0:
                return cutoff_y - 1
            return int(search_top + rows.max())

        left_bottom = _lowest_hair_y(
            int(x1 - face_w * 0.34),
            int(cx - face_w * 0.06),
        )
        right_bottom = _lowest_hair_y(
            int(cx + face_w * 0.06),
            int(x2 + face_w * 0.34),
        )
        bottom_excess = max(0, left_bottom - allowed_bottom) + max(0, right_bottom - allowed_bottom)
        bottom_score = float(
            np.clip(1.0 - ((0.5 * bottom_excess) / max(face_h * 0.32, 1.0)), 0.0, 1.0)
        )

        if left_bottom >= cutoff_y and right_bottom >= cutoff_y:
            balance_score = float(
                np.clip(1.0 - (abs(left_bottom - right_bottom) / max(face_h * 0.24, 1.0)), 0.0, 1.0)
            )
        else:
            balance_score = 0.0

        side_top = max(0, int(y1 - face_h * 0.04))
        side_bottom = min(H, int(cutoff_y + face_h * 0.16))

        def _band_coverage(xa: int, xb: int) -> float:
            xa = max(0, min(xa, W))
            xb = max(0, min(xb, W))
            if xa >= xb or side_top >= side_bottom:
                return 0.0
            band = hair_u8[side_top:side_bottom, xa:xb] > 0
            if band.size == 0:
                return 0.0
            return float(band.mean())

        left_coverage = _band_coverage(
            int(x1 - face_w * 0.30),
            int(x1 + face_w * 0.14),
        )
        right_coverage = _band_coverage(
            int(x2 - face_w * 0.14),
            int(x2 + face_w * 0.30),
        )
        presence_score = float(
            np.clip((min(left_coverage, right_coverage) - 0.03) / 0.10, 0.0, 1.0)
        )

        area_ratio = float(short_real_px / max(face_area, 1.0))
        if area_ratio < 0.58:
            area_score = float(np.clip(area_ratio / 0.58, 0.0, 1.0))
        elif area_ratio > 1.95:
            area_score = float(np.clip(1.0 - ((area_ratio - 1.95) / 1.10), 0.0, 1.0))
        else:
            area_score = 1.0

        score = float(np.clip(
            remnant_score * 0.36
            + neck_clear_score * 0.20
            + bottom_score * 0.14
            + presence_score * 0.12
            + balance_score * 0.08
            + area_score * 0.10,
            0.0,
            1.0,
        ))
        return {
            "score": score,
            "remnant_score": remnant_score,
            "neck_clear_score": neck_clear_score,
            "bottom_score": bottom_score,
            "presence_score": presence_score,
            "balance_score": balance_score,
            "area_score": area_score,
            "remnant_ratio": remnant_ratio,
            "neck_occupancy": neck_occupancy,
            "left_presence": float(left_coverage),
            "right_presence": float(right_coverage),
            "left_bottom_px": float(left_bottom),
            "right_bottom_px": float(right_bottom),
            "allowed_bottom_px": float(allowed_bottom),
            "short_area_ratio": area_ratio,
            "short_real_mask_pixels": float(short_real_px),
            "short_remnant_pixels": float(remnant_px),
            "original_long_pixels": float(long_px),
        }

    def _preserve_original_hair_tone(
        self,
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        src_hair, _, _ = self._segface_hair_mask(source_rgb, face_bbox)
        tgt_hair, _, _ = self._segface_hair_mask(target_rgb, face_bbox)

        src_mask = (src_hair > 0.45)
        tgt_mask = (tgt_hair > 0.45)
        if int(src_mask.sum()) < 100 or int(tgt_mask.sum()) < 100:
            return target_rgb

        src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        src_mean = src_lab[src_mask].mean(axis=0)
        tgt_mean = tgt_lab[tgt_mask].mean(axis=0)

        tuned_lab = tgt_lab.copy()
        vals = tuned_lab[tgt_mask]
        vals[:, 1] = np.clip(vals[:, 1] + (src_mean[1] - tgt_mean[1]) * 0.78, 0.0, 255.0)
        vals[:, 2] = np.clip(vals[:, 2] + (src_mean[2] - tgt_mean[2]) * 0.78, 0.0, 255.0)
        vals[:, 0] = np.clip(vals[:, 0] + (src_mean[0] - tgt_mean[0]) * 0.32, 0.0, 255.0)
        tuned_lab[tgt_mask] = vals

        tuned_rgb = cv2.cvtColor(tuned_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        alpha = cv2.GaussianBlur(tgt_hair.astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=3.0)
        alpha = np.clip(alpha * 0.70, 0.0, 1.0)[..., np.newaxis]
        out = tuned_rgb.astype(np.float32) * alpha + target_rgb.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        hairstyle_text: str,
        color_text: str,
        hair_length: str = "long",
    ) -> Tuple[str, str, float]:
        """
        Returns:
            positive_prompt, negative_prompt, guidance_scale
        """
        normalized_color = MirrAISDPipeline._normalize_color_text(color_text)
        lowered_style = (hairstyle_text or "").lower()
        explicit_flyaway_request = any(
            token in lowered_style
            for token in ("flyaway", "flyaways", "wispy", "잔머리", "뱅", "bang")
        )
        parts = []
        if hairstyle_text:
            parts.append(hairstyle_text.strip())
        if normalized_color:
            parts.append(f"{normalized_color.strip()} hair color")
        style = ", ".join(parts) if parts else "natural hairstyle"

        # ── 길이별 positive/negative 보강 ────────────────────────────────────
        if hair_length == "short":
            wispy_suffix = (
                "soft wispy texture only near bangs and jawline tips, no dangling strands below jawline"
                if explicit_flyaway_request
                else "clean outer contour with softly feathered ends, no loose hanging tendrils below jawline"
            )
            pos_suffix = (
                ", short chin-length bob, soft layered ends, feathered tapered tips, "
                "natural uneven hairline near jaw, clear neckline, visible neck, "
                "hair ends stop around jawline, does not touch shoulders, "
                f"mostly above jawline, {wispy_suffix}"
            )
            neg_prefix = (
                "very long hair, flowing long hair, hair below shoulders, "
                "hair below chin, hair touching shoulders, hair covering chest, "
                "waist-length hair, side long locks over chest, "
                "center-parted long front panels, curtain-like long side pieces, "
                "long face-framing layers below jawline, elongated front sections, "
                "dangling front tendrils, face-framing strands below jawline, "
                "wispy long strands on neck, flyaway strands below jawline, "
                "thin hanging side pieces, stringy strands on chest, "
                "blunt horizontal cut line, helmet hair, bowl-shaped edge, "
                "earrings, earring, dangling earrings, hoop earrings, pearl earrings, "
                "jewelry, necklace, pendant, choker, accessories, piercings, "
            )
            guidance = 10.8
        elif hair_length == "medium":
            pos_suffix = (
                ", medium length hair, shoulder-length hair, "
                "hair just above or at shoulder"
            )
            neg_prefix = "very long hair, very short hair, "
            guidance = 8.5
        else:
            pos_suffix = ""
            neg_prefix = ""
            guidance = 7.5

        color_pos_hint = ""
        color_neg_hint = ""
        lowered_color = normalized_color.lower()
        if "ash" in lowered_color:
            color_pos_hint = ", cool-toned ash color, smoky neutral undertone, no brassiness"
            color_neg_hint = "warm orange cast, yellow brassiness, copper tint, reddish tint, "
        elif normalized_color:
            color_pos_hint = ", consistent natural hair color tone, coherent root-to-end color"

        positive_parts = [
            f"professional portrait photo of a person with {style}{pos_suffix}",
        ]
        if color_pos_hint:
            positive_parts.append(color_pos_hint.lstrip(", ").strip())
        positive_parts.extend([
            "photorealistic, high quality, natural lighting, 8k",
            "studio photography, sharp focus, beautiful hair",
        ])
        positive = ", ".join(positive_parts)
        negative_base = _NEGATIVE_BASE
        negative = neg_prefix + color_neg_hint + negative_base

        return positive, negative, guidance

    # ──────────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate(
        self,
        img_512: Image.Image,
        mask_512: Image.Image,
        canny_512: Image.Image,
        face_crop_pil: Image.Image,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        seeds: List[int],
        hair_length: str = "long",
    ) -> List[Image.Image]:
        """
        모든 seed를 단일 배치 forward pass로 생성 (순차 대비 ~절반 시간).

        diffusers는 generator를 리스트로 받으면 num_images_per_prompt 개의
        이미지를 각자 다른 seed로 한 번의 파이프라인 실행에 처리함.
        """
        # 숏컷/중단발 변환 시 IP-Adapter scale을 낮춤
        # → 원본 긴머리 identity가 생성에 과도하게 영향주는 것 방지
        if hair_length == "short":
            ip_scale = 0.0
            control_scale = min(self.config.controlnet_conditioning_scale, 0.05)
        elif hair_length == "medium":
            ip_scale = 0.18
            control_scale = min(self.config.controlnet_conditioning_scale, 0.20)
        else:
            ip_scale = self.config.ip_adapter_scale  # long은 기본값 유지
            control_scale = self.config.controlnet_conditioning_scale

        self._sd_pipe.set_ip_adapter_scale(ip_scale)
        logger.info(
            f"[SDPipeline] ip_adapter_scale={ip_scale}, "
            f"controlnet_scale={control_scale} (hair_length={hair_length})"
        )

        n = len(seeds)
        generators = [
            torch.Generator(device=self.device).manual_seed(s) for s in seeds
        ]
        logger.info(f"[SDPipeline] 배치 생성 시작 (n={n}, seeds={seeds})")

        with torch.inference_mode():
            out = self._sd_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_512,
                mask_image=mask_512,
                control_image=canny_512,
                ip_adapter_image=[face_crop_pil],
                height=SD_SIZE,
                width=SD_SIZE,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=control_scale,
                num_images_per_prompt=n,
                generator=generators,
                strength=1.0,
            )

        logger.info(f"[SDPipeline] 배치 생성 완료 → {len(out.images)}장")
        return out.images

    def _sd_refine_removed_region(
        self,
        base_rgb: np.ndarray,          # H×W×3 RGB (cv2 inpaint 1차 결과)
        removal_mask: np.ndarray,      # H×W float32 (긴머리 제거 영역)
        face_bbox: Tuple[int, int, int, int],
        face_crop_pil: Image.Image,    # IP-Adapter conditioning face
        protect_mask: Optional[np.ndarray],  # H×W float32 (얼굴 보호)
        cloth_mask: Optional[np.ndarray],    # H×W float32 (의상 영역)
        hair_length: str,
        seed: int,
    ) -> np.ndarray:
        """
        긴머리 제거 후 남는 어색한 영역(목/어깨/배경)을 SD로 한 번 더 정리.
        """
        H, W = base_rgb.shape[:2]
        if removal_mask.shape != (H, W):
            raise ValueError(f"removal_mask shape mismatch: {removal_mask.shape} vs {(H, W)}")

        # removal 영역 중심으로만 SD를 적용하기 위해 그대로 letterbox 변환
        fill_mask = (removal_mask > 0.5).astype(np.float32)
        img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
            base_rgb,
            fill_mask,
            mask_edge_suppression=0.45,
        )

        if hair_length == "short":
            fill_prompt = (
                "professional portrait photo, clean natural neck and shoulders, "
                "realistic clothing fabric texture continuity, coherent background, "
                "short-hair silhouette maintained, no long hair below jawline, "
                "no loose dangling strands in masked region, photorealistic details"
            )
            fill_guidance = 7.1
        else:
            fill_prompt = (
                "professional portrait photo, clean neck and shoulders, "
                "natural skin and clothing texture continuity, coherent background, "
                "no loose long hair strands in masked region, photorealistic details"
            )
            fill_guidance = 7.6
        fill_negative = (
            "long hair, hair below chin, hair below shoulders, loose hair strands, "
            "wavy hair, straight long hair, wig, ponytail, braid, bangs, side locks, "
            "earrings, earring, dangling earrings, hoop earrings, pearl earrings, "
            "jewelry, necklace, pendant, choker, accessories, piercings, "
            "deformed neck, artifacts, blurry, smudged texture, melted details, cartoon, painting"
        )

        # 배경 복원은 identity 영향이 과하면 긴머리가 다시 생길 수 있어 scale을 낮춘다.
        self._sd_pipe.set_ip_adapter_scale(0.0)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.18), 0.12, 0.30))
        fill_steps = max(24, self.config.num_inference_steps - 4)

        with torch.inference_mode():
            out = self._sd_pipe(
                prompt=fill_prompt,
                negative_prompt=fill_negative,
                image=img_512,
                mask_image=mask_512,
                control_image=canny_512,
                ip_adapter_image=[face_crop_pil],
                height=SD_SIZE,
                width=SD_SIZE,
                num_inference_steps=fill_steps,
                guidance_scale=fill_guidance,
                controlnet_conditioning_scale=fill_control,
                num_images_per_prompt=1,
                generator=generator,
                strength=0.88,
            )

        gen_np = np.array(out.images[0])  # 512×512 RGB

        # letterbox 역변환
        pad_l, pad_t = pad
        new_w = int(W * scale)
        new_h = int(H * scale)
        gen_cropped = gen_np[pad_t:pad_t + new_h, pad_l:pad_l + new_w]
        gen_orig = cv2.resize(gen_cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

        alpha = cv2.GaussianBlur(fill_mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        # 중앙 편향을 완화해 side 잔존 영역도 자연스럽게 복원한다.
        x1, y1, x2, y2 = face_bbox
        cx = 0.5 * (x1 + x2)
        face_w = max(float(x2 - x1), 1.0)
        sigma_x = max(face_w * 1.45, 44.0)
        xs = np.arange(W, dtype=np.float32)
        center_weight = np.exp(-0.5 * ((xs - cx) / sigma_x) ** 2)
        alpha = alpha * (0.65 + 0.35 * center_weight[np.newaxis, :])

        # 의상 영역은 과도한 hallucination을 줄이기 위해 SD 블렌딩 가중치를 낮춘다.
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_w = np.clip(cloth_mask.astype(np.float32), 0.0, 1.0)
            alpha = alpha * (1.0 - 0.18 * cloth_w)

        # 얼굴은 기존 픽셀 고정
        if protect_mask is not None:
            protect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            protect = cv2.dilate(protect_mask.astype(np.float32), protect_k)
            alpha = alpha * (1.0 - np.clip(protect, 0.0, 1.0))

        alpha = alpha[..., np.newaxis]
        refined = (
            gen_orig.astype(np.float32) * alpha
            + base_rgb.astype(np.float32) * (1.0 - alpha)
        )
        return np.clip(refined, 0, 255).astype(np.uint8)

    def _build_preclean_mask_for_two_step(
        self,
        removal_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        cloth_mask: Optional[np.ndarray],
        hair_length: str,
    ) -> np.ndarray:
        """
        two-step pre-clean용 확장 마스크 생성.
        하이브리드 short/medium 입력에서 하단 긴머리만 부분 선철거하도록
        removal_mask를 적당히 확장하되, 목/가슴 중앙은 과도하게 먹지 않게 제한한다.
        """
        H, W = removal_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        base_u8 = (np.clip(removal_mask, 0.0, 1.0) > 0.35).astype(np.uint8) * 255
        if int((base_u8 > 0).sum()) < 40:
            return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

        # 기존 full pre-clean보다 훨씬 좁게 확장한다.
        kx = max(11, int(face_w * (0.26 if hair_length == "short" else 0.28)))
        ky = max(13, int(face_h * (0.24 if hair_length == "short" else 0.26)))
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
        preclean_u8 = cv2.dilate(base_u8, dilate_k, iterations=1)

        # cutoff 근처 아래쪽만 허용
        top_y = max(0, int(cutoff_y - face_h * (0.05 if hair_length == "short" else 0.09)))
        top_gate = np.zeros((H, W), dtype=np.uint8)
        top_gate[top_y:, :] = 255
        preclean_u8 = cv2.bitwise_and(preclean_u8, top_gate)

        # 얼굴 주변 side corridor 안에서만 유지
        corridor_x_ratio = 1.28 if hair_length == "short" else 1.35
        x_min = max(0, int(x1 - face_w * corridor_x_ratio))
        x_max = min(W, int(x2 + face_w * corridor_x_ratio))
        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        if x_min < x_max:
            corridor_u8[:, x_min:x_max] = 255
            preclean_u8 = cv2.bitwise_and(preclean_u8, corridor_u8)

        # 옷 경계 쪽 잔머리는 조금 더 포함하되, side corridor 안에서만 넓힌다.
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_u8 = (np.clip(cloth_mask, 0.0, 1.0) > 0.18).astype(np.uint8) * 255
            cloth_u8 = cv2.bitwise_and(cloth_u8, corridor_u8)
            cloth_u8[:top_y, :] = 0
            if int((cloth_u8 > 0).sum()) > 20:
                cloth_band = cv2.dilate(
                    cloth_u8,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (21, 21) if hair_length == "short" else (21, 21),
                    ),
                    iterations=1,
                )
                base_support = cv2.dilate(
                    base_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                    iterations=1,
                )
                cloth_band = cv2.bitwise_and(cloth_band, base_support)
                preclean_u8 = cv2.bitwise_or(preclean_u8, cloth_band)

        # neck/upper-chest band에서는 좌우 hair blob 사이를 메워
        # 중앙 아래로 내려온 긴머리 스트랜드도 같이 지운다.
        bridge_bottom = min(H, int(cutoff_y + face_h * (0.84 if hair_length == "short" else 0.82)))
        bridge_fill = np.zeros((H, W), dtype=np.uint8)
        gap_limit = int(face_w * (0.86 if hair_length == "short" else 0.86))
        for row_y in range(top_y, bridge_bottom):
            row = preclean_u8[row_y, :] > 0
            left = np.flatnonzero(row[:cx])
            right = np.flatnonzero(row[cx:])
            if left.size == 0 or right.size == 0:
                continue
            left_inner = int(left.max())
            right_inner = int(cx + right.min())
            if 0 < (right_inner - left_inner) <= gap_limit:
                bridge_fill[row_y, left_inner:right_inner + 1] = 255
        if int((bridge_fill > 0).sum()) > 0:
            bridge_fill = cv2.dilate(
                bridge_fill,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
            preclean_u8 = cv2.bitwise_or(preclean_u8, bridge_fill)

        # torso 중앙 보호는 더 아래쪽에서만 얇게 적용한다.
        center_half = max(10, int(face_w * (0.11 if hair_length == "short" else 0.18)))
        center_top = min(H, int(cutoff_y + face_h * (0.94 if hair_length == "short" else 0.90)))
        center_bottom = min(H, int(cutoff_y + face_h * (1.36 if hair_length == "short" else 1.42)))
        center_cut = np.zeros((H, W), dtype=np.uint8)
        center_cut[
            center_top:center_bottom,
            max(0, cx - center_half):min(W, cx + center_half),
        ] = 255
        preclean_u8 = cv2.bitwise_and(preclean_u8, cv2.bitwise_not(center_cut))

        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        preclean_u8 = cv2.morphologyEx(preclean_u8, cv2.MORPH_CLOSE, close_k)

        max_ratio = 0.22 if hair_length == "short" else 0.24
        max_px = int(H * W * max_ratio)
        cur_px = int((preclean_u8 > 0).sum())
        if cur_px > max_px:
            shrink_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            for _ in range(6):
                preclean_u8 = cv2.erode(preclean_u8, shrink_k, iterations=1)
                cur_px = int((preclean_u8 > 0).sum())
                if cur_px <= max_px:
                    break

        preclean = (preclean_u8 > 0).astype(np.float32)
        return np.clip(preclean, 0.0, 1.0).astype(np.float32)

    def _sd_preclean_long_hair_region(
        self,
        base_rgb: np.ndarray,
        preclean_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        face_crop_pil: Image.Image,
        protect_mask: Optional[np.ndarray],
        hair_length: str,
        seed: int,
    ) -> np.ndarray:
        """
        Two-step pre-clean pass.
        긴머리 흔적을 먼저 지운 뒤(묶은 머리/올백 컨셉), 2차 헤어 생성으로 넘긴다.
        """
        H, W = base_rgb.shape[:2]
        if preclean_mask.shape != (H, W):
            return base_rgb

        fill_mask = (np.clip(preclean_mask, 0.0, 1.0) > 0.45).astype(np.float32)
        if int((fill_mask > 0).sum()) < 80:
            return base_rgb

        fill_edge_suppress = 0.58 if hair_length == "short" else 0.32
        img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
            base_rgb,
            fill_mask,
            mask_edge_suppression=fill_edge_suppress,
            canny_suppress_mask=fill_mask,
        )

        if hair_length == "short":
            clean_prompt = (
                "professional portrait photo, tightly tied-back slicked-back hair silhouette, "
                "clean exposed neck and shoulder line, bare collar area in masked region, "
                "no dangling side hair strands, no visible hanging strand over clothing, "
                "no hair below jawline in masked region, clean jawline contour, "
                "no thin tendrils crossing neck, no flyaway strands below jawline, "
                "coherent sweater neckline and clothing texture, "
                "photorealistic details"
            )
        else:
            clean_prompt = (
                "professional portrait photo, tied-back or slicked-back hair silhouette, "
                "clean neck and shoulder area, no loose long strands in masked region, "
                "no hair below shoulder line in masked region, coherent background and clothing texture, "
                "photorealistic details"
            )
        clean_negative = (
            "long hanging hair, side locks over chest, loose strands, visible ponytail, braid, "
            "single long strand on shoulder, dark strand on clothing, dangling strand by neck, "
            "face-framing tendrils below jawline, thin wispy strand crossing neck, "
            "stringy flyaways below chin, long sideburn strand on collar, "
            "hat, cap, beanie, helmet, hairnet, headscarf, bandana, head covering, "
            "wavy long hair, hair below shoulders, messy flyaway clumps, wig-like texture, "
            "artifacts, blurred texture, melted details, cartoon, painting"
        )

        self._sd_pipe.set_ip_adapter_scale(0.0)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        if hair_length == "short":
            clean_control = float(np.clip(self.config.controlnet_conditioning_scale * 0.22, 0.04, 0.09))
            clean_steps = max(28, self.config.num_inference_steps)
            clean_strength = float(np.clip(max(self.config.preclean_strength, 0.985), 0.94, 0.995))
            clean_guidance = 7.9
        else:
            clean_control = float(np.clip(self.config.controlnet_conditioning_scale * 0.45, 0.08, 0.16))
            clean_steps = max(26, self.config.num_inference_steps - 2)
            clean_strength = float(np.clip(self.config.preclean_strength, 0.86, 0.99))
            clean_guidance = 7.3

        with torch.inference_mode():
            out = self._sd_pipe(
                prompt=clean_prompt,
                negative_prompt=clean_negative,
                image=img_512,
                mask_image=mask_512,
                control_image=canny_512,
                ip_adapter_image=[face_crop_pil],
                height=SD_SIZE,
                width=SD_SIZE,
                num_inference_steps=clean_steps,
                guidance_scale=clean_guidance,
                controlnet_conditioning_scale=clean_control,
                num_images_per_prompt=1,
                generator=generator,
                strength=clean_strength,
            )

        gen_np = np.array(out.images[0])  # 512x512 RGB
        pad_l, pad_t = pad
        new_w = int(W * scale)
        new_h = int(H * scale)
        gen_cropped = gen_np[pad_t:pad_t + new_h, pad_l:pad_l + new_w]
        gen_orig = cv2.resize(gen_cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

        alpha = cv2.GaussianBlur(fill_mask, (0, 0), sigmaX=9.0, sigmaY=9.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        if protect_mask is not None and protect_mask.shape == (H, W):
            protect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            protect = cv2.dilate(np.clip(protect_mask, 0.0, 1.0), protect_k, iterations=1)
            alpha = alpha * (1.0 - np.clip(protect, 0.0, 1.0))

        alpha = alpha[..., np.newaxis]
        precleaned = gen_orig.astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
        return np.clip(precleaned, 0, 255).astype(np.uint8)

    def _build_real_short_remnant_mask(
        self,
        img_rgb: np.ndarray,
        original_long_hair_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        protect_mask: Optional[np.ndarray] = None,
        hair_length: str = "short",
        return_debug: bool = False,
        current_hair_mask: Optional[np.ndarray] = None,
    ) -> Any:
        """
        실제 생성 결과에서 short hair mask를 다시 추출한 뒤,
        M_remnant = M_long - M_short_real 을 계산한다.
        이후 remnant를 dilate하여 LaMa cleanup 경계 artifact를 줄인다.
        """
        H, W = img_rgb.shape[:2]
        if original_long_hair_mask.shape != (H, W):
            empty = np.zeros((H, W), dtype=np.float32)
            if return_debug:
                return empty, empty, {
                    "short_real_mask_pixels": 0,
                    "short_remnant_pixels": 0,
                }
            return empty, empty

        _, y1, _, y2 = face_bbox
        face_h = max(int(y2 - y1), 1)

        if current_hair_mask is None:
            current_hair_mask, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
        if protect_mask is not None and protect_mask.shape == (H, W):
            current_hair_mask = np.clip(current_hair_mask - np.clip(protect_mask, 0.0, 1.0), 0.0, 1.0)

        # 실제 short silhouette로 인정할 하단 범위를 제한한다.
        keep_bottom = min(
            H - 1,
            int(cutoff_y + face_h * (0.10 if hair_length == "short" else 0.26)),
        )
        short_real = current_hair_mask.copy()
        short_real[keep_bottom + 1 :, :] = 0.0

        short_real_u8 = (short_real > 0.45).astype(np.uint8) * 255
        short_real_u8 = cv2.morphologyEx(
            short_real_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

        long_u8 = (np.clip(original_long_hair_mask, 0.0, 1.0) > 0.35).astype(np.uint8) * 255
        remnant_u8 = cv2.bitwise_and(long_u8, cv2.bitwise_not(short_real_u8))
        remnant_u8[:cutoff_y, :] = 0

        if protect_mask is not None and protect_mask.shape == (H, W):
            protect_u8 = (np.clip(protect_mask, 0.0, 1.0) > 0.35).astype(np.uint8) * 255
            remnant_u8 = cv2.bitwise_and(remnant_u8, cv2.bitwise_not(protect_u8))

        remnant_u8 = cv2.morphologyEx(
            remnant_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        remnant_u8 = cv2.dilate(
            remnant_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 9) if hair_length == "short" else (7, 7),
            ),
            iterations=1,
        )

        short_real_mask = short_real_u8.astype(np.float32) / 255.0
        remnant_mask = remnant_u8.astype(np.float32) / 255.0
        if return_debug:
            return remnant_mask, short_real_mask, {
                "short_real_mask_pixels": int((short_real_u8 > 0).sum()),
                "short_remnant_pixels": int((remnant_u8 > 0).sum()),
            }
        return remnant_mask, short_real_mask

    def _build_shoulder_protect_mask(
        self,
        cloth_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
    ) -> np.ndarray:
        """
        어깨선(옷 상단 경계) 보호 마스크 생성.
        short/medium 후처리에서 어깨 라인 훼손을 줄이기 위해 사용한다.
        """
        H, W = cloth_mask.shape[:2]
        if cloth_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        cloth_u8 = (cloth_mask > 0.35).astype(np.uint8) * 255
        if int((cloth_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        y_start = max(0, int(cutoff_y - face_h * 0.08))
        y_end = min(H, int(cutoff_y + face_h * 1.05))
        band = np.zeros((H, W), dtype=np.uint8)
        band[y_start:y_end, :] = cloth_u8[y_start:y_end, :]
        if int((band > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        edge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge = cv2.morphologyEx(band, cv2.MORPH_GRADIENT, edge_k)
        edge = cv2.dilate(
            edge,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )

        x_min = max(0, int(x1 - face_w * 1.30))
        x_max = min(W, int(x2 + face_w * 1.30))
        if x_min >= x_max:
            return np.zeros((H, W), dtype=np.float32)

        corridor = np.zeros((H, W), dtype=np.uint8)
        corridor[:, x_min:x_max] = 255
        edge = cv2.bitwise_and(edge, corridor)

        center_half = max(18, int(face_w * 0.45))
        center_zone = np.zeros((H, W), dtype=np.uint8)
        center_zone[:, max(0, cx - center_half):min(W, cx + center_half)] = 255
        side_edge = cv2.bitwise_and(edge, cv2.bitwise_not(center_zone))
        if int((side_edge > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        side_edge = cv2.GaussianBlur(
            side_edge.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=3.0,
            sigmaY=3.0,
        )
        return np.clip(side_edge, 0.0, 1.0).astype(np.float32)

    def _remove_residual_hair_below_cutoff(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        shoulder_protect: Optional[np.ndarray] = None,
        hair_length: str = "short",
        return_debug: bool = False,
        debug_prefix: str = "residual",
        min_pixels: int = 60,
        open_kernel_size: int = 5,
        dilate_kernel_size: int = 9,
        top_offset_px: int = 0,
    ) -> Any:
        """
        short/medium 변환 후 cutoff 아래에 남은 머리카락을 재검출해 정리.
        """
        H, W = img_rgb.shape[:2]
        cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
        _, y1, _, y2 = face_bbox
        face_h = max(int(y2 - y1), 1)
        cleanup_start_y = int(np.clip(cutoff_y + int(top_offset_px), 0, H - 1))
        soft_zone = max(10, int(face_h * 0.22))
        soft_end = min(H - 1, cleanup_start_y + soft_zone)

        hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
        residual = hair_now.copy()
        residual[:cleanup_start_y, :] = 0.0

        # 작은 노이즈 제거
        residual_u8 = (residual > 0.5).astype(np.uint8) * 255
        if open_kernel_size % 2 == 0:
            open_kernel_size += 1
        if dilate_kernel_size % 2 == 0:
            dilate_kernel_size += 1
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_OPEN, open_k)

        if shoulder_protect is not None and shoulder_protect.shape == (H, W):
            protect_threshold = 0.84 if hair_length == "short" else 0.34
            protect_u8 = (shoulder_protect > protect_threshold).astype(np.uint8) * 255
            if hair_length == "short" and int((protect_u8 > 0).sum()) > 0:
                protect_u8 = cv2.erode(
                    protect_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                )
            if int((protect_u8 > 0).sum()) > 0:
                residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(protect_u8))

        # cutoff 바로 아래는 완만히 제거해 단발 끝선이 일자로 잘린 느낌을 완화
        if soft_end > cleanup_start_y:
            ramp = np.ones((H,), dtype=np.float32)
            ramp[:cleanup_start_y] = 0.0
            ramp[cleanup_start_y:soft_end + 1] = np.linspace(
                0.0, 1.0, soft_end - cleanup_start_y + 1, dtype=np.float32
            )
            residual_soft = (residual_u8.astype(np.float32) / 255.0) * ramp[:, np.newaxis]
            residual_u8 = (residual_soft > 0.50).astype(np.uint8) * 255

        residual_mask = residual_u8.astype(np.float32) / 255.0
        residual_pixels = int((residual_u8 > 0).sum())
        mask_key = f"pipeline_lama_{debug_prefix}_mask"
        result_key = f"pipeline_lama_{debug_prefix}_result"
        applied_key = f"lama_{debug_prefix}_applied"
        pixels_key = f"lama_{debug_prefix}_pixels"
        if residual_pixels < int(min_pixels):
            if return_debug:
                return img_rgb, {
                    mask_key: residual_mask,
                    applied_key: False,
                    pixels_key: residual_pixels,
                }
            return img_rgb

        dilate_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_kernel_size, dilate_kernel_size),
        )
        residual_u8 = cv2.dilate(residual_u8, dilate_k, iterations=1)
        residual_mask = residual_u8.astype(np.float32) / 255.0

        # LaMa로 잔존 hair 영역 제거
        result_rgb = self._lama_inpaint(img_rgb, residual_u8, force_single_pass=True)
        if return_debug:
            return result_rgb, {
                mask_key: residual_mask,
                result_key: result_rgb,
                applied_key: True,
                pixels_key: residual_pixels,
            }
        return result_rgb

    def _final_cutoff_cleanup(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        removal_mask: np.ndarray,
        cutoff_y: int,
        shoulder_protect: Optional[np.ndarray] = None,
        hair_length: str = "short",
        current_hair_mask: Optional[np.ndarray] = None,
        return_debug: bool = False,
    ) -> Any:
        """
        최종 결과에서 cutoff 아래 long-hair 제거 마스크 영역을 한 번 더 정리.
        """
        H, W = img_rgb.shape[:2]
        if removal_mask.shape != (H, W):
            if return_debug:
                return img_rgb, {
                    "pipeline_lama_post_cleanup_mask": np.zeros((H, W), dtype=np.float32),
                    "lama_post_cleanup_applied": False,
                }
            return img_rgb

        x1, y1, x2, y2 = face_bbox
        face_h = max(int(y2 - y1), 1)
        face_w = max(int(x2 - x1), 1)
        cx = int(0.5 * (x1 + x2))
        soft_zone = max(12, int(face_h * 0.25))

        force = removal_mask.copy().astype(np.float32)
        cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
        force[:cutoff_y, :] = 0.0
        soft_end = min(H - 1, cutoff_y + soft_zone)
        if soft_end > cutoff_y:
            ramp = np.ones((H,), dtype=np.float32)
            ramp[:cutoff_y] = 0.0
            ramp[cutoff_y:soft_end + 1] = np.linspace(
                0.0, 1.0, soft_end - cutoff_y + 1, dtype=np.float32
            )
            force = force * ramp[:, np.newaxis]

        # 얼굴 주변 corridor 안에서만 cleanup을 허용해 의상/배경 훼손을 줄인다.
        corridor_ratio = 1.55 if hair_length == "short" else 1.35
        x_min = max(0, int(x1 - face_w * corridor_ratio))
        x_max = min(W, int(x2 + face_w * corridor_ratio))
        corridor = np.zeros((H, W), dtype=np.uint8)
        if x_min < x_max:
            corridor[:, x_min:x_max] = 255

        force_thresh = 0.56 if hair_length == "short" else 0.54
        force_u8 = ((force > force_thresh).astype(np.uint8) * 255)
        force_u8 = cv2.bitwise_and(force_u8, corridor)

        # 실제 남아있는 hair 픽셀과 교집합을 우선 적용해 의상/배경 훼손 방지
        if current_hair_mask is not None and current_hair_mask.shape == (H, W):
            hair_now = np.clip(current_hair_mask.astype(np.float32), 0.0, 1.0)
        else:
            hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
        hair_now[:cutoff_y, :] = 0.0
        hair_now_u8 = (hair_now > 0.5).astype(np.uint8) * 255
        if int((hair_now_u8 > 0).sum()) > 0:
            hair_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (11, 11) if hair_length == "short" else (9, 9),
            )
            hair_now_u8 = cv2.dilate(hair_now_u8, hair_k, iterations=1)

        hair_inter_u8 = cv2.bitwise_and(force_u8, hair_now_u8)
        if hair_length == "short":
            # SegFace miss를 보완하기 위해 side-zone에 한해 high-confidence force를 추가 반영
            center_half = max(18, int(face_w * 0.42))
            side_zone = corridor.copy()
            side_zone[:, max(0, cx - center_half):min(W, cx + center_half)] = 0
            side_fallback_u8 = ((force > 0.78).astype(np.uint8) * 255)
            side_fallback_u8 = cv2.bitwise_and(side_fallback_u8, side_zone)

            # 중심부의 가는 세로 스트랜드는 SegFace miss가 잦다.
            # 턱선보다 충분히 아래에서만 보수적으로 fallback을 허용한다.
            center_force_u8 = ((force > 0.70).astype(np.uint8) * 255)
            center_zone = np.zeros((H, W), dtype=np.uint8)
            center_top = min(H, int(cutoff_y + face_h * 0.18))
            center_bottom = min(H, int(cutoff_y + face_h * 1.45))
            center_x_half = max(16, int(face_w * 0.24))
            center_zone[
                center_top:center_bottom,
                max(0, cx - center_x_half):min(W, cx + center_x_half),
            ] = 255
            center_force_u8 = cv2.bitwise_and(center_force_u8, center_zone)

            center_fallback_u8 = np.zeros_like(center_force_u8)
            if int((center_force_u8 > 0).sum()) > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    center_force_u8,
                    connectivity=8,
                )
                min_center_area = max(18, int(face_w * 0.10))
                min_center_height = max(16, int(face_h * 0.18))
                for label_idx in range(1, num_labels):
                    x, y, w, h, area = stats[label_idx]
                    if area < min_center_area:
                        continue
                    if h < min_center_height:
                        continue
                    center_fallback_u8[labels == label_idx] = 255
                if int((center_fallback_u8 > 0).sum()) > 0:
                    center_fallback_u8 = cv2.dilate(
                        center_fallback_u8,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
                        iterations=1,
                    )

            force_u8 = cv2.bitwise_or(hair_inter_u8, side_fallback_u8)
            force_u8 = cv2.bitwise_or(force_u8, center_fallback_u8)
        else:
            # medium도 SegFace miss 보완용 fallback force 일부 허용
            fallback_u8 = ((force > 0.74).astype(np.uint8) * 255)
            fallback_u8 = cv2.bitwise_and(fallback_u8, corridor)
            force_u8 = cv2.bitwise_or(hair_inter_u8, fallback_u8)

        if shoulder_protect is not None and shoulder_protect.shape == (H, W):
            protect_threshold = 0.84 if hair_length == "short" else 0.34
            protect_u8 = (shoulder_protect > protect_threshold).astype(np.uint8) * 255
            if hair_length == "short" and int((protect_u8 > 0).sum()) > 0:
                protect_u8 = cv2.erode(
                    protect_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                )
            if int((protect_u8 > 0).sum()) > 0:
                force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(protect_u8))

        min_cleanup_px = 28 if hair_length == "short" else 40
        force_mask = force_u8.astype(np.float32) / 255.0
        force_pixels = int((force_u8 > 0).sum())
        if int((force_u8 > 0).sum()) < min_cleanup_px:
            if return_debug:
                return img_rgb, {
                    "pipeline_lama_post_cleanup_mask": force_mask,
                    "lama_post_cleanup_applied": False,
                    "lama_post_cleanup_pixels": force_pixels,
                }
            return img_rgb

        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9) if hair_length == "short" else (7, 7),
        )
        force_u8 = cv2.dilate(force_u8, k, iterations=1)
        force_mask = force_u8.astype(np.float32) / 255.0

        # LaMa로 잔존 hair 제거
        result_rgb = self._lama_inpaint(img_rgb, force_u8, force_single_pass=True)
        if return_debug:
            return result_rgb, {
                "pipeline_lama_post_cleanup_mask": force_mask,
                "pipeline_lama_post_cleanup_result": result_rgb,
                "lama_post_cleanup_applied": True,
                "lama_post_cleanup_pixels": force_pixels,
            }
        return result_rgb

    # ──────────────────────────────────────────────────────────────────────────
    # Compositing
    # ──────────────────────────────────────────────────────────────────────────

    def _composite(
        self,
        orig_bgr: np.ndarray,
        orig_rgb: np.ndarray,
        gen_pil: Image.Image,            # 512×512 RGB
        hair_mask: np.ndarray,           # H×W float32 (original resolution)
        scale: float,
        pad: Tuple[int, int],            # (pad_left, pad_top)
        original_size: Tuple[int, int],  # (W, H)
        protect_mask: Optional[np.ndarray] = None,  # H×W float32: 이 영역은 alpha=0 강제 (얼굴 보호)
        hair_length: str = "long",
    ) -> np.ndarray:
        """
        SD 생성 이미지를 원본에 합성.
        - hair mask 영역: SD 생성 결과
        - 그 외 (+ protect_mask): 원본 (얼굴/배경 유지)
        """
        W, H = original_size
        pad_l, pad_t = pad
        new_w = int(W * scale)
        new_h = int(H * scale)

        if orig_rgb.shape[:2] != (H, W):
            orig_rgb = self._ensure_rgb_image_shape(
                orig_rgb,
                (H, W),
                context="composite_orig_rgb",
            )
        if hair_mask.shape != (H, W):
            logger.warning(
                f"[SDPipeline] composite hair_mask shape mismatch → resize: "
                f"got={hair_mask.shape}, expected={(H, W)}"
            )
            hair_mask = cv2.resize(
                hair_mask.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_LINEAR,
            )

        # letterbox 제거 → 원본 비율로 crop
        gen_np = np.array(gen_pil)   # 512×512×3 RGB
        gen_cropped = gen_np[pad_t:pad_t + new_h, pad_l:pad_l + new_w]

        # 원본 해상도로 upscale
        gen_orig = cv2.resize(gen_cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

        # alpha 블렌딩: short/medium는 경계를 더 또렷하게 유지
        sigma = 6.0
        if hair_length == "short":
            sigma = 4.2
        elif hair_length == "medium":
            sigma = 4.8
        alpha = cv2.GaussianBlur(hair_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if hair_length == "short":
            alpha = np.clip((alpha - 0.10) / 0.90, 0.0, 1.0)
        elif hair_length == "medium":
            alpha = np.clip((alpha - 0.07) / 0.93, 0.0, 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        # 얼굴/귀/눈 등 보호 영역: alpha를 0으로 강제
        # → Gaussian blur가 얼굴 경계로 번지더라도 원본 픽셀 100% 유지
        if protect_mask is not None:
            # protect_mask도 살짝 dilate해서 경계까지 확실히 보호
            protect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            protect_dilated = cv2.dilate(protect_mask.astype(np.float32), protect_k)
            alpha = alpha * (1.0 - np.clip(protect_dilated, 0.0, 1.0))

        alpha = alpha[..., np.newaxis]   # H×W×1

        orig_f = orig_rgb.astype(np.float32)
        gen_f  = gen_orig.astype(np.float32)
        blend  = gen_f * alpha + orig_f * (1.0 - alpha)
        blend  = np.clip(blend, 0, 255).astype(np.uint8)

        return cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def unload(self) -> None:
        """VRAM 해제"""
        import gc
        self._sd_pipe = None
        self._bisenet = None
        self._sam2_factory = None
        if self._mp_face:
            self._mp_face.close()
        if self._mp_face_mesh:
            self._mp_face_mesh.close()
        self._mp_face = None
        self._mp_face_mesh = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[SDPipeline] 모델 언로드 완료")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 테스트
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True)
    parser.add_argument("--hairstyle", default="wolf cut, layered")
    parser.add_argument("--color",     default="auburn")
    parser.add_argument("--top-k",     type=int, default=3)
    parser.add_argument("--output",    default="./sd_output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    pipe = MirrAISDPipeline()
    pipe.load()

    results = pipe.run(img, args.hairstyle, args.color, args.top_k)

    os.makedirs(args.output, exist_ok=True)
    for r in results:
        path = os.path.join(args.output, f"rank{r.rank}_seed{r.seed}_{r.mask_used}.jpg")
        cv2.imwrite(path, r.image)
        print(f"저장: {path}")
