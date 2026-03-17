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
_BALD_HAIR_KEYWORDS = frozenset([
    "bald", "shaved head", "clean scalp", "skinhead", "buzz 0", "buzzcut 0",
    "삭발", "대머리", "민머리",
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

    # 메모리 최적화
    enable_xformers: bool = True

    # 후처리 옵션 (현재 파이프라인에서는 기본 alpha blend 사용)
    use_clip_ranking: bool = False   # 향후 CLIP 랭킹 확장용
    use_color_match:  bool = False   # 향후 LAB 색상 매칭 확장용
    use_poisson_blend: bool = False  # 향후 Poisson blend 확장용

    # 배경 채우기 모드 (short/medium 변환 시 긴머리 제거 방법)
    #   "cv2" : cv2.inpaint(NS+TELEA) 블렌딩 (기본, 빠름)
    #   "sd"  : cv2 1차 + SD 복원 보정 2차 (품질↑, 시간↑)
    bg_fill_mode: str = "cv2"

    # short/medium 2-step 전략:
    #   step-1: long hair 흔적 제거(pre-clean, tied/slicked back 컨셉)
    #   step-2: target hairstyle 생성
    enable_two_step_preclean: bool = True
    preclean_mask_expand_ratio_x: float = 1.65
    preclean_mask_expand_ratio_y: float = 1.00
    preclean_strength: float = 0.96


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
    clip_score: float = 0.0 # CLIP 점수 (현재는 rank 순서, 향후 CLIP 랭킹 확장용)
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
        self._sam2_factory = None  # SAM2 predictor factory (callable)
        self._sd_pipe    = None   # StableDiffusionControlNetInpaintPipeline
        self._mp_face    = None   # MediaPipe FaceDetection
        self._mp_face_mesh = None # MediaPipe FaceMesh
        self._mp_hands   = None   # MediaPipe Hands
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
        self._load_sam2()
        self._load_mediapipe()
        self._load_sd_pipeline()
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

        if debug_images_common is not None:
            debug_images_common["pipeline_input_image"] = image.copy()

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
        hand_mask = self._detect_hand_mask(img_rgb)
        if hand_mask.sum() > 0:
            logger.info(f"[SDPipeline] 손 보호 마스크 검출: pixels={hand_mask.sum():.0f}")
        _store_mask("mediapipe_hand_mask", hand_mask)

        # ── Step 2: SegFace base hair mask + 얼굴 픽셀 마스크 + 옷 마스크 ──────
        hair_mask_base, face_region_mask, cloth_mask = self._segface_hair_mask(img_rgb, face_bbox)
        _store_mask("segface_hair_mask", hair_mask_base)
        _store_mask("segface_face_region_mask", face_region_mask)
        _store_mask("segface_cloth_mask", cloth_mask)

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
        is_bald_style = self._is_bald_request(hairstyle_text)
        hair_length = self._classify_hair_length(hairstyle_text)
        logger.info(f"[SDPipeline] 헤어 길이 분류: {hair_length} (bald={is_bald_style})")

        # ── Step 3-c: SegFace 얼굴 픽셀 제거 (bbox 직사각형 대신 픽셀 단위 보정) ─
        hair_mask = np.clip(hair_mask - face_region_mask, 0.0, 1.0)
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

        # ── Step 3-e: 숏컷/중단발 — 2단계 전략 ──────────────────────────────
        # 단발/숏컷 변환은 단순 SD inpainting으로 불가능:
        #   SD는 마스크 주변에 긴 머리가 보이면 계속 긴 머리를 생성함.
        # 해결: ① cv2.inpaint로 긴 머리 먼저 제거(배경/피부로 채움)
        #        ② 제거된 이미지 위에 SD로 새 숏컷 생성 (머리 없는 주변 맥락)
        cutoff_y_for_post: Optional[int] = None
        removal_mask_for_post: Optional[np.ndarray] = None
        shoulder_protect_for_post: Optional[np.ndarray] = None
        if hair_length in ("short", "medium"):
            x1f, y1f, x2f, y2f = face_bbox
            face_w = max(x2f - x1f, 1)
            face_h = max(y2f - y1f, 1)

            if hair_length == "short":
                cutoff_y = int(y2f + face_h * 0.12)   # 턱~윗목 사이 (하드컷 완화)
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

            # ── head box 계산 (removal_mask, gen_mask 양쪽에서 사용) ─────────────
            margin_x = int(face_w * 0.5)   # 얼굴 폭의 50% 여백 (양 옆 머리 공간)
            head_x1  = max(0, x1f - margin_x)
            head_x2  = min(W, x2f + margin_x)
            head_y1  = max(0, y1f - int(face_h * 0.6))   # 정수리 위까지
            head_y2  = cutoff_y

            # ── 제거 마스크: SAM2 감지 영역 + 어깨 너비 확장 ────────────────────
            # SAM2가 한쪽 머리를 놓친 경우(오른쪽 등) 를 커버하기 위해
            # cutoff_y 아래 전체 어깨 너비를 removal 영역으로 포함
            # short에서는 SAM2 과검출(손/소품/배경 포함) 억제를 위해
            # SegFace 기반 hair prior와 교집합을 우선 사용한다.
            if hair_length == "short":
                base_prior = np.clip(hair_mask_base - face_region_mask, 0.0, 1.0).astype(np.float32)
                prior_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
                base_prior = cv2.dilate((base_prior > 0.5).astype(np.uint8) * 255, prior_k, iterations=1)
                base_prior = (base_prior > 0).astype(np.float32)

                removal_seed = np.clip(hair_mask_for_removal * base_prior, 0.0, 1.0)
                # short 변환에서는 cutoff 아래 long-tail은 적극적으로 제거한다.
                # (prior 교집합이 너무 보수적으로 작동할 때 하단 잔존 모발이 남는 문제 보완)
                tail_below_cutoff = hair_mask_for_removal.copy()
                tail_below_cutoff[:cutoff_y, :] = 0.0
                tail_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 9))
                tail_below_cutoff = cv2.dilate(
                    (tail_below_cutoff > 0.5).astype(np.uint8) * 255,
                    tail_k,
                    iterations=1,
                ).astype(np.float32) / 255.0

                # 교집합이 과도하게 작으면 기존 SAM 기반 마스크로 폴백
                if removal_seed.sum() > 120:
                    removal_mask = np.clip(np.maximum(removal_seed, tail_below_cutoff), 0.0, 1.0)
                else:
                    removal_mask = hair_mask_for_removal.copy()
            else:
                # medium은 의상 훼손이 크므로 cloth-protected hair mask를 기본으로 사용.
                removal_mask = hair_mask.copy()
            removal_mask[:cutoff_y, :] = 0.0

            # 실제 hair 픽셀 기반으로만 확장 (직사각형 전체 확장 시 옷 패치 훼손 유발)
            hair_below_src = hair_mask_for_removal if hair_length == "short" else hair_mask
            hair_below = (hair_below_src > 0.5).astype(np.uint8)
            hair_below[:cutoff_y, :] = 0
            hair_below[:, :max(0, head_x1 - 20)] = 0
            hair_below[:, min(W, head_x2 + 20):] = 0
            if int((hair_below > 0).sum()) > 0:
                # short는 연결 보정만, medium은 어깨선까지 조금 더 강하게 확장
                expand_k = (
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 9))
                    if hair_length == "short"
                    else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 13))
                )
                hair_below_expanded = cv2.dilate(hair_below, expand_k, iterations=1).astype(np.float32)
                removal_mask = np.maximum(removal_mask, hair_below_expanded)
                logger.info(
                    f"[SDPipeline] removal_mask hair기반 확장({hair_length}): "
                    f"pixels={removal_mask.sum():.0f}"
                )

                # SegFace/SAM miss로 중앙이 끊겨 있으면 neck corridor에서 좌우를 연결.
                # (잔존 장발 + 생성 헤어 이중 경계 완화)
                bridge_u8 = np.zeros((H, W), dtype=np.uint8)
                bridge_src_u8 = (hair_below_expanded > 0).astype(np.uint8)
                bridge_y2 = min(H, int(cutoff_y + face_h * (1.10 if hair_length == "short" else 1.45)))
                max_span = int(face_w * (2.70 if hair_length == "short" else 3.00))
                for row in range(cutoff_y, bridge_y2):
                    cols = np.where(bridge_src_u8[row] > 0)[0]
                    if cols.size < 2:
                        continue
                    c_min = int(cols.min())
                    c_max = int(cols.max())
                    if c_max - c_min > max_span:
                        continue
                    bridge_u8[row, c_min:c_max + 1] = 255
                if int((bridge_u8 > 0).sum()) > 0:
                    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))
                    bridge_u8 = cv2.morphologyEx(bridge_u8, cv2.MORPH_CLOSE, smooth_k)
                    bridge_zone = np.zeros((H, W), dtype=np.uint8)
                    bridge_zone[cutoff_y:bridge_y2, max(0, head_x1 - 14):min(W, head_x2 + 14)] = 255
                    bridge_u8 = cv2.bitwise_and(bridge_u8, bridge_zone)

                support_k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (111, 47) if hair_length == "short" else (131, 55),
                )
                support_u8 = cv2.dilate((hair_below > 0).astype(np.uint8) * 255, support_k, iterations=1)
                bridge_u8 = cv2.bitwise_and(bridge_u8, support_u8)

                if int((bridge_u8 > 0).sum()) > 0:
                    removal_mask = np.maximum(removal_mask, bridge_u8.astype(np.float32) / 255.0)
                    logger.info(
                        f"[SDPipeline] removal_mask 연결 보강({hair_length}): "
                        f"pixels={int((bridge_u8 > 0).sum())}"
                    )
            # short는 최소 연결만, medium은 조금 더 강하게 연결
            close_size = 5 if hair_length == "short" else 11
            remove_close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            removal_mask = cv2.morphologyEx(removal_mask, cv2.MORPH_CLOSE, remove_close_k)
            removal_mask = np.clip(removal_mask, 0.0, 1.0).astype(np.float32)
            removal_mask_for_post = removal_mask.copy()

            # ── 생성 마스크: head box 기반 (아치형만이 아닌 머리 전체 공간) ─────
            # 기존: hair_mask의 cutoff_y 위쪽 (아치형만 → 귀 옆 공간 없음)
            # 변경: face_bbox 기반 head box → 귀 옆까지 포함해서 SD가 숏컷 생성

            if is_bald_style:
                # bald: SAM2 full hair mask 기반으로 전체 머리 영역을 커버해야 함.
                # hair_mask_base(SegFace)는 너무 작아서 양옆 머리를 놓침.
                # hair_mask_for_removal = SAM2 refined mask (face 제거 완료 상태)
                bald_u8 = (hair_mask_for_removal > 0.3).astype(np.uint8) * 255
                bald_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
                bald_u8 = cv2.dilate(bald_u8, bald_k, iterations=1)
                # 이마 + 정수리 영역도 포함 (머리카락이 없더라도 두피 연속성 필요)
                forehead_u8 = np.zeros((H, W), dtype=np.uint8)
                fx1 = max(0, int(x1f - face_w * 0.35))
                fx2 = min(W, int(x2f + face_w * 0.35))
                fy1 = max(0, int(y1f - face_h * 0.55))
                fy2 = min(H, int(y1f + face_h * 0.10))
                if fx1 < fx2 and fy1 < fy2:
                    forehead_u8[fy1:fy2, fx1:fx2] = 255
                bald_mask = np.maximum(bald_u8.astype(np.float32) / 255.0, forehead_u8.astype(np.float32) / 255.0)
                # 얼굴 내부는 보호하되 이마 위쪽 헤어라인은 포함
                gen_mask = np.clip(bald_mask - face_region_mask * 0.72, 0.0, 1.0)
                # 옷 보호 (bald는 머리만 바꾸고 옷은 절대 건드리면 안 됨)
                gen_mask = np.clip(gen_mask - cloth_mask_dilated, 0.0, 1.0)
                gen_mask = np.clip(gen_mask - hand_mask, 0.0, 1.0)
            else:
                gen_mask = np.zeros((H, W), dtype=np.float32)
                gen_mask[head_y1:head_y2, head_x1:head_x2] = 1.0
                # 얼굴 내부(눈/코/입)는 inpaint 하지 않음
                gen_mask = np.clip(gen_mask - face_region_mask, 0.0, 1.0)
                # 옷도 보호
                if hair_length == "medium":
                    # medium은 끝선 생성을 위해 의상 보호를 완화(완전 차단시 하단 notching 발생)
                    gen_mask = np.clip(gen_mask - cloth_mask_dilated * 0.55, 0.0, 1.0)
                else:
                    gen_mask = np.clip(gen_mask - cloth_mask_dilated, 0.0, 1.0)
                # 손/소품도 보호
                gen_mask = np.clip(gen_mask - hand_mask, 0.0, 1.0)

            # 제거 단계에서도 손/소품은 보호
            removal_mask = np.clip(removal_mask - hand_mask, 0.0, 1.0)
            # 어깨 윤곽(옷 상단 경계)은 과도하게 지우지 않도록 보호
            if shoulder_protect_for_post is not None:
                shoulder_protect_weight = 0.40 if hair_length == "short" else 0.50
                removal_mask = np.clip(
                    removal_mask - shoulder_protect_for_post * shoulder_protect_weight,
                    0.0,
                    1.0,
                )
            _store_mask("pipeline_short_removal_mask", removal_mask)
            _store_mask("pipeline_short_generation_mask", gen_mask)

            logger.info(
                f"[SDPipeline] 2단계 숏컷: removal_px={removal_mask.sum():.0f}, "
                f"gen_px={gen_mask.sum():.0f}, cutoff_y={cutoff_y}, "
                f"head_box=({head_x1},{head_y1})-({head_x2},{head_y2})"
            )

            # 단계 1: 긴 머리 제거 ─────────────────────────────────────────────
            bg_mode = self.config.bg_fill_mode
            logger.info(f"[SDPipeline] bg_fill_mode={bg_mode}")

            if removal_mask.sum() > 50:
                removal_u8 = (removal_mask > 0.5).astype(np.uint8) * 255
                # inpaint 입력은 약하게만 확장해서 불필요한 배경/의상 훼손을 줄임
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                removal_u8_dilated = cv2.dilate(removal_u8, k, iterations=1)

                # cv2.inpaint 2종(NS + TELEA) 혼합 → 어깨/목 texture 복원 안정화
                inpaint_radius = 8 if hair_length == "short" else 10
                img_rgb_ns = cv2.inpaint(
                    img_rgb, removal_u8_dilated, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS
                )
                img_rgb_telea = cv2.inpaint(
                    img_rgb, removal_u8_dilated, inpaintRadius=max(3, inpaint_radius), flags=cv2.INPAINT_TELEA
                )
                img_rgb_filled = cv2.addWeighted(img_rgb_ns, 0.25, img_rgb_telea, 0.75, 0.0)

                # 작은/중간 영역은 seamlessClone으로 경계 접합을 자연스럽게 처리
                # (분리된 마스크에서는 왜곡/패치가 커서 alpha blend로 폴백)
                removal_px = int((removal_u8 > 0).sum())
                max_clone_px = int(H * W * 0.18)
                cc_count = 0
                largest_cc_ratio = 0.0
                if removal_px > 0:
                    cc_num, _, cc_stats, _ = cv2.connectedComponentsWithStats(removal_u8, connectivity=8)
                    cc_count = max(0, int(cc_num - 1))
                    if cc_count > 0:
                        largest_cc = int(cc_stats[1:, cv2.CC_STAT_AREA].max())
                        largest_cc_ratio = largest_cc / float(removal_px)
                allow_clone = (
                    50 <= removal_px <= max_clone_px
                    and cc_count == 1
                    and largest_cc_ratio >= 0.82
                )
                # short/medium 헤어 제거에서는 seamlessClone이 목/어깨 패치를 만드는 케이스가 많아 기본 비활성화.
                allow_clone = False
                if allow_clone:
                    ys, xs = np.where(removal_u8 > 0)
                    c_x = int((xs.min() + xs.max()) * 0.5)
                    c_y = int((ys.min() + ys.max()) * 0.5)
                    src_bgr = cv2.cvtColor(img_rgb_filled, cv2.COLOR_RGB2BGR)
                    dst_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    try:
                        cloned_bgr = cv2.seamlessClone(
                            src_bgr, dst_bgr, removal_u8, (c_x, c_y), cv2.NORMAL_CLONE
                        )
                        img_rgb_cleaned = cv2.cvtColor(cloned_bgr, cv2.COLOR_BGR2RGB)
                    except Exception:
                        # fallback: alpha blend
                        soft_alpha = removal_u8.astype(np.float32) / 255.0
                        soft_alpha = cv2.GaussianBlur(soft_alpha, (0, 0), sigmaX=1.2)[..., np.newaxis]
                        soft_alpha = np.clip((soft_alpha - 0.28) / 0.72, 0.0, 1.0)
                        img_rgb_cleaned = (
                            img_rgb_filled.astype(np.float32) * soft_alpha
                            + img_rgb.astype(np.float32) * (1.0 - soft_alpha)
                        ).clip(0, 255).astype(np.uint8)
                else:
                    if cc_count > 1:
                        logger.info(
                            "[SDPipeline] seamlessClone 스킵: 분리 마스크 "
                            f"(components={cc_count}, largest_ratio={largest_cc_ratio:.3f})"
                        )
                    # soft-blend는 core mask 기준으로만 feather → 사다리꼴 번짐 방지
                    soft_alpha = removal_u8.astype(np.float32) / 255.0
                    soft_alpha = cv2.GaussianBlur(soft_alpha, (0, 0), sigmaX=1.2)[..., np.newaxis]
                    soft_alpha = np.clip((soft_alpha - 0.28) / 0.72, 0.0, 1.0)
                    img_rgb_cleaned = (
                        img_rgb_filled.astype(np.float32) * soft_alpha
                        + img_rgb.astype(np.float32) * (1.0 - soft_alpha)
                    ).clip(0, 255).astype(np.uint8)

                # 옷(어깨/상체)과 겹친 제거 영역은 별도 복원해서
                # 반투명 얼룩/패치가 남는 문제를 완화한다.
                cloth_u8 = (cloth_mask_dilated > 0.45).astype(np.uint8) * 255
                cloth_overlap = cv2.bitwise_and(removal_u8, cloth_u8)
                protect_u8 = (((face_region_mask + hand_mask) > 0.2).astype(np.uint8) * 255)
                cloth_overlap = cv2.bitwise_and(cloth_overlap, cv2.bitwise_not(protect_u8))
                if int((cloth_overlap > 0).sum()) >= 80:
                    cloth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    cloth_overlap = cv2.dilate(cloth_overlap, cloth_k, iterations=1)
                    cloth_ns = cv2.inpaint(img_rgb, cloth_overlap, inpaintRadius=4, flags=cv2.INPAINT_NS)
                    cloth_te = cv2.inpaint(img_rgb, cloth_overlap, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                    cloth_refill = cv2.addWeighted(cloth_ns, 0.40, cloth_te, 0.60, 0.0)
                    cloth_alpha = cloth_overlap.astype(np.float32) / 255.0
                    cloth_alpha = cv2.GaussianBlur(
                        cloth_alpha, (0, 0), sigmaX=2.2, sigmaY=2.2
                    )[..., np.newaxis]
                    cloth_alpha = np.clip(cloth_alpha * 0.92, 0.0, 1.0)
                    img_rgb_cleaned = (
                        cloth_refill.astype(np.float32) * cloth_alpha
                        + img_rgb_cleaned.astype(np.float32) * (1.0 - cloth_alpha)
                    ).clip(0, 255).astype(np.uint8)
                    logger.info(
                        f"[SDPipeline] cloth overlap 복원 적용: pixels={int((cloth_overlap > 0).sum())}"
                    )
                logger.info("[SDPipeline] cv2.inpaint 2-way 블렌딩 완료")
                _store_rgb("cv2_background_cleaned_rgb", img_rgb_cleaned)

                if bg_mode == "sd":
                    # SD 보정 패스: 제거 영역의 배경/목/어깨 텍스처를 더 자연스럽게 정리
                    try:
                        fill_seed = int(seeds[0]) if seeds else random.randint(0, 2**31 - 1)
                        face_crop_fill = self._crop_face(Image.fromarray(img_rgb), face_bbox)
                        # 분리 컴포넌트 제거 마스크는 cv2 inpaint 흔적이 커서 원본 기반으로 SD 채움.
                        fill_base = img_rgb if cc_count > 1 else img_rgb_cleaned

                        # Two-step pre-clean:
                        #   step-1) 긴머리 흔적을 tied/slicked-back 컨셉으로 먼저 정리
                        #   step-2) 제거 영역 배경/목/어깨를 한 번 더 정리
                        if self.config.enable_two_step_preclean:
                            preclean_seed_mask = removal_mask
                            if is_bald_style:
                                # bald: SAM2 full mask 사용 (hair_mask_base는 너무 작음)
                                preclean_seed_mask = np.maximum(preclean_seed_mask, hair_mask_for_removal)
                            preclean_mask = self._build_preclean_mask_for_two_step(
                                removal_mask=preclean_seed_mask,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y,
                                cloth_mask=cloth_mask_dilated,
                                hand_mask=hand_mask,
                                hair_length=hair_length,
                                is_bald_style=is_bald_style,
                            )
                            _store_mask("pipeline_preclean_mask", preclean_mask)
                            if is_bald_style:
                                # bald: 옷도 보호 (preclean이 옷/배경을 바꾸는 문제 방지)
                                preclean_protect = np.clip(face_region_mask + hand_mask + cloth_mask_dilated, 0.0, 1.0)
                            else:
                                preclean_protect = np.clip(face_region_mask + hand_mask, 0.0, 1.0)
                            fill_base = self._sd_preclean_long_hair_region(
                                base_rgb=fill_base,
                                preclean_mask=preclean_mask,
                                face_bbox=face_bbox,
                                face_crop_pil=face_crop_fill,
                                protect_mask=preclean_protect,
                                hair_length=hair_length,
                                is_bald_style=is_bald_style,
                                seed=fill_seed + 13,
                            )
                            _store_rgb("sd_preclean_rgb", fill_base)
                            logger.info("[SDPipeline] two-step preclean 완료")

                        img_rgb_cleaned = self._sd_refine_removed_region(
                            base_rgb=fill_base,
                            removal_mask=removal_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_fill,
                            protect_mask=face_region_mask,
                            cloth_mask=cloth_mask_dilated,
                            hair_length=hair_length,
                            seed=fill_seed,
                        )
                        logger.info("[SDPipeline] bg_fill_mode=sd: 제거 영역 SD 보정 완료")
                    except Exception as e:
                        logger.warning(f"[SDPipeline] bg_fill_mode=sd 실패, cv2 결과 사용: {e}")

                # SD/cv2 제거 단계 이후에도 아래쪽 장발이 남거나 다시 생길 수 있어
                # cutoff 아래 hair를 한 번 더 검출해서 얇게 정리한다.
                try:
                    img_rgb_cleaned = self._remove_residual_hair_below_cutoff(
                        img_rgb=img_rgb_cleaned,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        shoulder_protect=shoulder_protect_for_post,
                        hair_length=hair_length,
                    )
                except Exception as e:
                    logger.warning(f"[SDPipeline] residual hair 정리 실패(무시): {e}")
            else:
                img_rgb_cleaned = img_rgb

            # 단계 2: SD 생성은 head box 중심(위쪽)으로 제한해 배경/의상 훼손을 줄임.
            # 아래쪽 장발 잔존은 final cleanup에서 선택적으로 제거한다.
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
        img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
            img_rgb_for_sd, hair_mask_for_sd
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
            hairstyle_text, normalized_color_text, hair_length, is_bald_style=is_bald_style
        )
        logger.info(f"[SDPipeline] 프롬프트: {prompt}")
        logger.info(f"[SDPipeline] 네거티브: {neg_prompt}")
        logger.info(f"[SDPipeline] guidance_scale: {guidance}")

        # ── Step 7: SD Inpainting ─────────────────────────────────────────────
        gen_images = self._generate(
            img_512, mask_512, canny_512, face_crop_pil, prompt, neg_prompt, guidance, seeds,
            hair_length=hair_length,
            is_bald_style=is_bald_style,
        )

        # ── Step 8: Composite → 원본 해상도 ───────────────────────────────────
        # composite base: short/medium은 cv2.inpaint된 이미지 사용
        #   → 긴머리 제거된 상태에서 새 숏컷을 얹음
        composite_base_rgb = img_rgb_cleaned
        composite_base_bgr = cv2.cvtColor(composite_base_rgb, cv2.COLOR_RGB2BGR)

        candidates: List[Dict[str, Any]] = []
        for gen_idx, (gen_pil, seed) in enumerate(zip(gen_images, seeds)):
            gen_preview_bgr = cv2.cvtColor(np.array(gen_pil), cv2.COLOR_RGB2BGR)
            composited_bgr = self._composite(
                composite_base_bgr, composite_base_rgb,
                gen_pil, hair_mask_for_sd, scale, pad, (W, H),
                protect_mask=face_region_mask,   # 얼굴 영역 alpha 침범 방지
                hair_length=hair_length,
            )

            # 최종 결과 기준으로 cutoff 아래 장발 잔존이 있으면 한 번 더 정리
            if cutoff_y_for_post is not None:
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    post_rgb = self._remove_residual_hair_below_cutoff(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        shoulder_protect=shoulder_protect_for_post,
                        hair_length=hair_length,
                    )
                    composited_bgr = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"[SDPipeline] final residual hair 정리 실패(무시): {e}")

            # short/medium: 원본 long-hair 제거 마스크 기반으로 최종 하드 컷
            if cutoff_y_for_post is not None and removal_mask_for_post is not None:
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    post_rgb = self._final_cutoff_cleanup(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        removal_mask=removal_mask_for_post,
                        cutoff_y=cutoff_y_for_post,
                        shoulder_protect=shoulder_protect_for_post,
                        hair_length=hair_length,
                    )
                    composited_bgr = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"[SDPipeline] final cutoff cleanup 실패(무시): {e}")

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

            color_distance: Optional[float] = None
            color_score = 0.0
            if has_color_request and target_hair_lab is not None:
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    color_distance = self._estimate_hair_color_distance(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        target_lab=target_hair_lab,
                    )
                    if color_distance is not None:
                        color_score = float(np.clip(1.0 - (color_distance / 80.0), 0.0, 1.0))
                except Exception as e:
                    logger.warning(f"[SDPipeline] 색상 거리 계산 실패(무시): {e}")

            candidates.append({
                "seed": seed,
                "image_bgr": composited_bgr,
                "preview_bgr": gen_preview_bgr,
                "color_distance": color_distance,
                "color_score": color_score,
                "gen_idx": gen_idx,
            })

        if has_color_request and target_hair_lab is not None and len(candidates) > 1:
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
                clip_score=float(cand["color_score"]),
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
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )
        logger.info("[SDPipeline] MediaPipe FaceDetection/FaceMesh/Hands 로드 완료")

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

    def _detect_hand_mask(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        MediaPipe Hands 기반 손 영역 마스크(HxW float32).
        손에 들린 소품(휴대폰)까지 보호하기 위해 landmark hull을 여유 있게 dilate한다.
        """
        H, W = img_rgb.shape[:2]
        hand_mask = np.zeros((H, W), dtype=np.float32)
        if self._mp_hands is None:
            return hand_mask

        result = self._mp_hands.process(img_rgb)
        if not result.multi_hand_landmarks:
            return hand_mask

        for hand_lm in result.multi_hand_landmarks:
            pts = []
            for lm in hand_lm.landmark:
                x = int(np.clip(lm.x * W, 0, W - 1))
                y = int(np.clip(lm.y * H, 0, H - 1))
                pts.append([x, y])
            if len(pts) < 3:
                continue
            pts_np = np.asarray(pts, dtype=np.int32)
            hull = cv2.convexHull(pts_np)
            cv2.fillConvexPoly(hand_mask, hull, 1.0)

        if hand_mask.sum() > 0:
            # 손 주변 소품까지 보호하도록 넉넉히 확장
            hand_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
            hand_mask = cv2.dilate(hand_mask, hand_k, iterations=1)
            hand_mask = cv2.GaussianBlur(hand_mask, (0, 0), sigmaX=4.0)
            hand_mask = np.clip(hand_mask, 0.0, 1.0).astype(np.float32)

        return hand_mask

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

        hair_512  = (parsing == HAIR_CLASS_IDX).astype(np.float32)
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
    ) -> Tuple[Image.Image, Image.Image, Image.Image, float, Tuple[int, int]]:
        """
        Letter-box resize → 512×512.

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
    def _is_bald_request(hairstyle_text: str) -> bool:
        text = hairstyle_text.lower()
        for kw in _BALD_HAIR_KEYWORDS:
            if kw in text:
                return True
        return False

    @staticmethod
    def _classify_hair_length(hairstyle_text: str) -> str:
        """헤어스타일 텍스트 → 'short' | 'medium' | 'long'"""
        text = hairstyle_text.lower()
        for kw in _BALD_HAIR_KEYWORDS:
            if kw in text:
                return "short"
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
    ) -> Optional[float]:
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
        is_bald_style: bool = False,
    ) -> Tuple[str, str, float]:
        """
        Returns:
            positive_prompt, negative_prompt, guidance_scale
        """
        normalized_color = MirrAISDPipeline._normalize_color_text(color_text)
        parts = []
        if hairstyle_text:
            parts.append(hairstyle_text.strip())
        if normalized_color:
            parts.append(f"{normalized_color.strip()} hair color")
        style = ", ".join(parts) if parts else "natural hairstyle"

        # ── 길이별 positive/negative 보강 ────────────────────────────────────
        if is_bald_style:
            pos_suffix = (
                ", shaved head, bald scalp, clean scalp skin texture, "
                "no bangs, no side hair, no visible loose strands, no headwear"
            )
            neg_prefix = (
                "long hair, medium hair, short bob, pixie cut, bangs, fringe, "
                "side locks, strands over forehead, hair over ears, "
                "hat, cap, beanie, helmet, hairnet, headscarf, bandana, head covering, "
            )
            guidance = 9.8
        elif hair_length == "short":
            pos_suffix = (
                ", short chin-length bob, soft layered ends, feathered tapered tips, "
                "natural uneven hairline near jaw, clear neckline, visible neck, "
                "mostly above jawline with a few natural wispy strands"
            )
            neg_prefix = (
                "very long hair, flowing long hair, hair below shoulders, "
                "waist-length hair, side long locks over chest, "
                "blunt horizontal cut line, helmet hair, bowl-shaped edge, "
                "earrings, earring, dangling earrings, hoop earrings, pearl earrings, "
                "jewelry, necklace, pendant, choker, accessories, piercings, "
            )
            guidance = 9.4
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
        if is_bald_style:
            negative_base = negative_base.replace("bald patch, ", "")
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
        is_bald_style: bool = False,
    ) -> List[Image.Image]:
        """
        모든 seed를 단일 배치 forward pass로 생성 (순차 대비 ~절반 시간).

        diffusers는 generator를 리스트로 받으면 num_images_per_prompt 개의
        이미지를 각자 다른 seed로 한 번의 파이프라인 실행에 처리함.
        """
        # 숏컷/중단발 변환 시 IP-Adapter scale을 낮춤
        # → 원본 긴머리 identity가 생성에 과도하게 영향주는 것 방지
        if hair_length == "short":
            if is_bald_style:
                ip_scale = 0.0
                control_scale = min(self.config.controlnet_conditioning_scale, 0.06)
            else:
                ip_scale = 0.05
                control_scale = min(self.config.controlnet_conditioning_scale, 0.12)
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
        hand_mask: Optional[np.ndarray],
        hair_length: str,
        is_bald_style: bool = False,
    ) -> np.ndarray:
        """
        two-step pre-clean용 확장 마스크 생성.
        긴머리 영역 + 어깨/배경 일부까지 넉넉하게 포함해 1차 제거 품질을 높인다.
        """
        H, W = removal_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)

        base_u8 = (np.clip(removal_mask, 0.0, 1.0) > 0.35).astype(np.uint8) * 255
        if int((base_u8 > 0).sum()) < 40:
            return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

        kx = max(21, int(face_w * float(self.config.preclean_mask_expand_ratio_x)))
        ky = max(13, int(face_h * float(self.config.preclean_mask_expand_ratio_y)))
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
        preclean_u8 = cv2.dilate(base_u8, dilate_k, iterations=1)

        if is_bald_style:
            # bald: SAM2 hair mask 기반이므로 dilated mask만으로 충분.
            # corridor를 추가하면 배경/옷까지 덮어서 SD가 전체를 재생성하는 문제 발생.
            # → corridor 없이 dilated hair mask만 사용.
            x_min, x_max, y_min, y_max = 0, 0, 0, 0  # corridor 비활성화
        else:
            corridor_x_ratio = 1.95 if hair_length == "short" else 2.15
            x_min = max(0, int(x1 - face_w * corridor_x_ratio))
            x_max = min(W, int(x2 + face_w * corridor_x_ratio))
            y_min = max(0, int(cutoff_y - face_h * 0.16))
            y_max = min(H, int(cutoff_y + face_h * (1.35 if hair_length == "short" else 1.65)))
        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        if x_min < x_max and y_min < y_max:
            corridor_u8[y_min:y_max, x_min:x_max] = 255
            preclean_u8 = cv2.bitwise_or(preclean_u8, corridor_u8)

        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_u8 = (np.clip(cloth_mask, 0.0, 1.0) > 0.12).astype(np.uint8) * 255
            cloth_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (31, 31) if hair_length == "short" else (37, 37),
            )
            cloth_u8 = cv2.dilate(cloth_u8, cloth_k, iterations=1)
            if is_bald_style:
                # bald: 옷 영역은 preclean에서 제외 (옷이 바뀌는 문제 방지)
                preclean_u8 = cv2.bitwise_and(preclean_u8, cv2.bitwise_not(cloth_u8))
            else:
                shoulder_band = cv2.bitwise_and(cloth_u8, corridor_u8)
                preclean_u8 = cv2.bitwise_or(preclean_u8, shoulder_band)

        if hand_mask is not None and hand_mask.shape == (H, W):
            hand_u8 = (np.clip(hand_mask, 0.0, 1.0) > 0.22).astype(np.uint8) * 255
            preclean_u8 = cv2.bitwise_and(preclean_u8, cv2.bitwise_not(hand_u8))

        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        preclean_u8 = cv2.morphologyEx(preclean_u8, cv2.MORPH_CLOSE, close_k)

        if is_bald_style:
            max_ratio = 0.38  # bald는 긴 머리 전체를 제거해야 하므로 넉넉하게
        else:
            max_ratio = 0.30 if hair_length == "short" else 0.34
        max_px = int(H * W * max_ratio)
        cur_px = int((preclean_u8 > 0).sum())
        if cur_px > max_px:
            shrink_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            for _ in range(5):
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
        is_bald_style: bool,
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

        img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
            base_rgb,
            fill_mask,
            mask_edge_suppression=0.22,
        )

        if is_bald_style:
            clean_prompt = (
                "professional portrait photo, clean bald scalp, bare smooth scalp skin texture, "
                "no bangs, no side hair, no visible strands, clean neck and shoulders, "
                "coherent background and clothing texture, photorealistic details"
            )
        elif hair_length == "short":
            clean_prompt = (
                "professional portrait photo, tightly tied-back slicked-back hair silhouette, "
                "clean exposed neck and shoulders, no dangling side hair strands, "
                "no hair below jawline in masked region, coherent background and clothing texture, "
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
            "hat, cap, beanie, helmet, hairnet, headscarf, bandana, head covering, "
            "wavy long hair, hair below shoulders, messy flyaway clumps, wig-like texture, "
            "artifacts, blurred texture, melted details, cartoon, painting"
        )

        self._sd_pipe.set_ip_adapter_scale(0.0)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        if is_bald_style:
            clean_control = float(np.clip(self.config.controlnet_conditioning_scale * 0.25, 0.04, 0.08))
        else:
            clean_control = float(np.clip(self.config.controlnet_conditioning_scale * 0.45, 0.08, 0.16))
        clean_steps = max(26, self.config.num_inference_steps - 2)
        clean_strength = float(np.clip(self.config.preclean_strength, 0.86, 0.99))

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
                guidance_scale=7.3,
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
    ) -> np.ndarray:
        """
        short/medium 변환 후 cutoff 아래에 남은 머리카락을 재검출해 정리.
        """
        H, W = img_rgb.shape[:2]
        cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
        _, y1, _, y2 = face_bbox
        face_h = max(int(y2 - y1), 1)
        soft_zone = max(10, int(face_h * 0.22))
        soft_end = min(H - 1, cutoff_y + soft_zone)

        hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
        residual = hair_now.copy()
        residual[:cutoff_y, :] = 0.0

        # 작은 노이즈 제거
        residual_u8 = (residual > 0.5).astype(np.uint8) * 255
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_OPEN, open_k)

        if shoulder_protect is not None and shoulder_protect.shape == (H, W):
            protect_threshold = 0.50 if hair_length == "short" else 0.34
            protect_u8 = (shoulder_protect > protect_threshold).astype(np.uint8) * 255
            if hair_length == "short" and int((protect_u8 > 0).sum()) > 0:
                protect_u8 = cv2.erode(
                    protect_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
            if int((protect_u8 > 0).sum()) > 0:
                residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(protect_u8))

        # cutoff 바로 아래는 완만히 제거해 단발 끝선이 일자로 잘린 느낌을 완화
        if soft_end > cutoff_y:
            ramp = np.ones((H,), dtype=np.float32)
            ramp[:cutoff_y] = 0.0
            ramp[cutoff_y:soft_end + 1] = np.linspace(
                0.0, 1.0, soft_end - cutoff_y + 1, dtype=np.float32
            )
            residual_soft = (residual_u8.astype(np.float32) / 255.0) * ramp[:, np.newaxis]
            residual_u8 = (residual_soft > 0.50).astype(np.uint8) * 255

        if int((residual_u8 > 0).sum()) < 60:
            return img_rgb

        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        residual_u8 = cv2.dilate(residual_u8, dilate_k, iterations=1)

        # 잔존 hair 영역만 얇게 inpaint
        filled_ns = cv2.inpaint(img_rgb, residual_u8, inpaintRadius=8, flags=cv2.INPAINT_NS)
        filled_te = cv2.inpaint(img_rgb, residual_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        filled = cv2.addWeighted(filled_ns, 0.6, filled_te, 0.4, 0.0)

        alpha = residual_u8.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=4.0, sigmaY=4.0)[..., np.newaxis]
        alpha = np.clip(alpha * 0.85, 0.0, 1.0)

        cleaned = (
            filled.astype(np.float32) * alpha
            + img_rgb.astype(np.float32) * (1.0 - alpha)
        )
        return np.clip(cleaned, 0, 255).astype(np.uint8)

    def _final_cutoff_cleanup(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        removal_mask: np.ndarray,
        cutoff_y: int,
        shoulder_protect: Optional[np.ndarray] = None,
        hair_length: str = "short",
    ) -> np.ndarray:
        """
        최종 결과에서 cutoff 아래 long-hair 제거 마스크 영역을 한 번 더 정리.
        """
        H, W = img_rgb.shape[:2]
        if removal_mask.shape != (H, W):
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
            fallback_u8 = ((force > 0.78).astype(np.uint8) * 255)
            fallback_u8 = cv2.bitwise_and(fallback_u8, side_zone)
            force_u8 = cv2.bitwise_or(hair_inter_u8, fallback_u8)
        else:
            # medium도 SegFace miss 보완용 fallback force 일부 허용
            fallback_u8 = ((force > 0.74).astype(np.uint8) * 255)
            fallback_u8 = cv2.bitwise_and(fallback_u8, corridor)
            force_u8 = cv2.bitwise_or(hair_inter_u8, fallback_u8)

        if shoulder_protect is not None and shoulder_protect.shape == (H, W):
            protect_threshold = 0.50 if hair_length == "short" else 0.34
            protect_u8 = (shoulder_protect > protect_threshold).astype(np.uint8) * 255
            if hair_length == "short" and int((protect_u8 > 0).sum()) > 0:
                protect_u8 = cv2.erode(
                    protect_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
            if int((protect_u8 > 0).sum()) > 0:
                force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(protect_u8))

        min_cleanup_px = 28 if hair_length == "short" else 40
        if int((force_u8 > 0).sum()) < min_cleanup_px:
            return img_rgb

        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9) if hair_length == "short" else (7, 7),
        )
        force_u8 = cv2.dilate(force_u8, k, iterations=1)

        inpaint_ns = 9 if hair_length == "short" else 8
        inpaint_te = 6 if hair_length == "short" else 5
        filled_ns = cv2.inpaint(img_rgb, force_u8, inpaintRadius=inpaint_ns, flags=cv2.INPAINT_NS)
        filled_te = cv2.inpaint(img_rgb, force_u8, inpaintRadius=inpaint_te, flags=cv2.INPAINT_TELEA)
        filled = cv2.addWeighted(filled_ns, 0.6, filled_te, 0.4, 0.0)

        alpha = force_u8.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=4.0, sigmaY=4.0)[..., np.newaxis]
        alpha_gain = 0.84 if hair_length == "short" else 0.78
        alpha = np.clip(alpha * alpha_gain, 0.0, 1.0)
        out = filled.astype(np.float32) * alpha + img_rgb.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

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
        if self._mp_hands:
            self._mp_hands.close()
        self._mp_face = None
        self._mp_face_mesh = None
        self._mp_hands = None
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
