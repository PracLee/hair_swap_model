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
from typing import List, Optional, Tuple

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

# ── BiSeNet 설정 ───────────────────────────────────────────────────────────────
HAIR_CLASS_IDX   = 10
BISENET_CLASSES  = 16
# 얼굴 내부 클래스 (skin/nose/eye_g/eyes/brows/ears/mouth/lips) - hair(10) 제거에 활용
FACE_CLASS_IDXS  = frozenset([1, 2, 3, 4, 5, 6, 7, 8, 9])

# ── SD 생성 해상도 ─────────────────────────────────────────────────────────────
SD_SIZE = 512   # SD 1.5 native resolution

# ── 공통 네거티브 프롬프트 ─────────────────────────────────────────────────────
_NEGATIVE_BASE = (
    "ugly, deformed, blurry, low quality, bad anatomy, distorted face, "
    "distorted hair, bald patch, artifacts, watermark, signature, "
    "cartoon, anime, illustration, painting, drawing"
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


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SDInpaintConfig:
    """SD Inpainting 파이프라인 설정"""
    # SD 생성 파라미터
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 0.5
    ip_adapter_scale: float = 0.6

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
    mask: Optional[np.ndarray] = None  # H×W float32 디버그용 마스크


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

        self._bisenet    = None   # BiSeNet face parsing
        self._sam2_factory = None  # SAM2 predictor factory (callable)
        self._sd_pipe    = None   # StableDiffusionControlNetInpaintPipeline
        self._mp_face    = None   # MediaPipe FaceDetection
        self._loaded     = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """모델 로드 (cold start). 이미 로드된 경우 no-op."""
        if self._loaded:
            return
        logger.info("[SDPipeline] 모델 로딩 시작...")
        self._load_bisenet()
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
    ) -> List[SDInpaintResult]:
        """
        헤어 스타일 변환 실행.

        Args:
            image:          입력 이미지 (BGR numpy)
            hairstyle_text: 헤어스타일 텍스트 (트렌드 데이터 hairstyle_text)
            color_text:     헤어 컬러 텍스트 (트렌드 데이터 color_text)
            top_k:          반환 결과 수 (기본 3)

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
        H, W = image.shape[:2]

        # ── Step 1: 얼굴 검출 ────────────────────────────────────────────────
        face_obs = self._detect_face(img_rgb)
        if face_obs is None:
            raise ValueError("얼굴을 검출할 수 없습니다.")
        face_bbox = face_obs  # (x1, y1, x2, y2)
        logger.info(f"[SDPipeline] 얼굴 검출: {face_bbox}")

        # ── Step 2: BiSeNet base hair mask + 얼굴 픽셀 마스크 ───────────────────
        hair_mask_base, face_region_mask = self._bisenet_hair_mask(img_rgb)

        # ── Step 3: SAM2 refinement ───────────────────────────────────────────
        hair_mask, mask_source = self._refine_with_sam2(
            img_rgb, hair_mask_base, face_bbox, hairstyle_text
        )
        logger.info(
            f"[SDPipeline] hair mask source={mask_source}, "
            f"pixels={hair_mask.sum():.0f}"
        )

        if hair_mask.sum() < 300:
            raise ValueError("머리카락 영역이 너무 작습니다.")

        # ── Step 3-b: 헤어 길이 분류 + 단발/숏컷일 때 마스크 하단 확장 ────────
        hair_length = self._classify_hair_length(hairstyle_text)
        logger.info(f"[SDPipeline] 헤어 길이 분류: {hair_length}")
        if hair_length in ("short", "medium"):
            hair_mask = self._expand_mask_for_short_hair(
                hair_mask, face_bbox, H, W, hair_length
            )
            logger.info(
                f"[SDPipeline] 마스크 하단 확장 완료 ({hair_length}), "
                f"pixels={hair_mask.sum():.0f}"
            )

        # ── Step 3-c: BiSeNet 얼굴 픽셀 제거 (bbox 직사각형 대신 픽셀 단위 보정) ─
        hair_mask = np.clip(hair_mask - face_region_mask, 0.0, 1.0)
        logger.info(
            f"[SDPipeline] 얼굴 픽셀 제거 완료, pixels={hair_mask.sum():.0f}"
        )

        # ── Step 4: SD 입력 준비 ─────────────────────────────────────────────
        img_pil = Image.fromarray(img_rgb)
        img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
            img_rgb, hair_mask
        )

        # ── Step 5: 얼굴 crop (IP-Adapter) ───────────────────────────────────
        face_crop_pil = self._crop_face(img_pil, face_bbox)

        # ── Step 6: 프롬프트 ─────────────────────────────────────────────────
        prompt, neg_prompt, guidance = self._build_prompt(
            hairstyle_text, color_text, hair_length
        )
        logger.info(f"[SDPipeline] 프롬프트: {prompt}")
        logger.info(f"[SDPipeline] 네거티브: {neg_prompt}")
        logger.info(f"[SDPipeline] guidance_scale: {guidance}")

        # ── Step 7: SD Inpainting ─────────────────────────────────────────────
        gen_images = self._generate(
            img_512, mask_512, canny_512, face_crop_pil, prompt, neg_prompt, guidance, seeds
        )

        # ── Step 8: Composite → 원본 해상도 ───────────────────────────────────
        results: List[SDInpaintResult] = []
        for rank, (gen_pil, seed) in enumerate(zip(gen_images, seeds)):
            composited_bgr = self._composite(
                image, img_rgb, gen_pil, hair_mask, scale, pad, (W, H)
            )
            results.append(SDInpaintResult(
                image=composited_bgr,
                image_pil=Image.fromarray(cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)),
                seed=seed,
                rank=rank,
                mask_used=mask_source,
                mask=hair_mask,
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
        logger.info("[SDPipeline] MediaPipe FaceDetection 로드 완료")

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

    # ──────────────────────────────────────────────────────────────────────────
    # Segmentation: BiSeNet + SAM2
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

    def _bisenet_hair_mask(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        BiSeNet으로 머리카락 + 얼굴 영역 마스크 생성.
        입력: H×W×3 RGB (any resolution)
        출력:
            hair_mask  H×W float32 [0, 1]
            face_mask  H×W float32 [0, 1]  (skin/눈/코/입 등 얼굴 내부 픽셀)
        """
        H, W = img_rgb.shape[:2]

        # BiSeNet 입력: 512×512, [-1, 1] 정규화
        inp_np = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
        inp_t = torch.from_numpy(inp_np).float().permute(2, 0, 1).unsqueeze(0)
        inp_t = inp_t / 127.5 - 1.0
        if self.dtype == torch.float16:
            inp_t = inp_t.half()
        inp_t = inp_t.to(self.device)

        with torch.no_grad():
            output = self._bisenet(inp_t)
            # BiSeNet.forward()는 (magnify(feat_out)[1024×1024], feat_out[256×256]) 튜플을 반환.
            # feat_out[1] 이 실제 분류 logit (입력의 1/2 해상도, AdaptiveAvgPool로 blurred되지 않음).
            if isinstance(output, (list, tuple)):
                logits = output[1]   # feat_out — 실제 분류 logit
            else:
                logits = output
            parsing = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        hair_512 = (parsing == HAIR_CLASS_IDX).astype(np.float32)
        face_512 = np.isin(parsing, list(FACE_CLASS_IDXS)).astype(np.float32)

        hair_orig = cv2.resize(hair_512, (W, H), interpolation=cv2.INTER_LINEAR)
        face_orig = cv2.resize(face_512, (W, H), interpolation=cv2.INTER_LINEAR)
        return (hair_orig > 0.5).astype(np.float32), (face_orig > 0.5).astype(np.float32)

    def _refine_with_sam2(
        self,
        img_rgb: np.ndarray,           # H×W×3 RGB
        base_mask: np.ndarray,          # H×W float32
        face_bbox: Tuple[int, int, int, int],
        prompt_text: str,
    ) -> Tuple[np.ndarray, str]:
        """
        SAM2로 BiSeNet 마스크를 정밀 보정.

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

            # SAM2 bbox: BiSeNet hair mask의 실제 범위로 tight하게 설정
            hair_coords = np.argwhere(base_mask > 0.5)  # (N, 2) [row, col]
            if len(hair_coords) > 0:
                hr_min, hc_min = hair_coords.min(axis=0)
                hr_max, hc_max = hair_coords.max(axis=0)
                sam_bbox = np.array([
                    max(0,     hc_min - 10),
                    max(0,     hr_min - 10),
                    min(W - 1, hc_max + 10),
                    min(H - 1, hr_max + 10),
                ], dtype=np.float32)
            else:
                sam_bbox = np.array([
                    max(0, x1 - int(bw * 0.2)),
                    max(0, y1 - int(bh * 0.5)),
                    min(W - 1, x2 + int(bw * 0.2)),
                    min(H - 1, y2 + int(bh * 0.1)),
                ], dtype=np.float32)

            # Positive points: BiSeNet hair 영역 상단 5점 (얼굴과 거리 먼 곳)
            if len(hair_coords) > 0:
                # 상단 30% 픽셀에서만 샘플링 → 얼굴과 겹치는 하단 제외
                top_cutoff = hr_min + int((hr_max - hr_min) * 0.3)
                top_coords = hair_coords[hair_coords[:, 0] <= top_cutoff]
                if len(top_coords) < 3:
                    top_coords = hair_coords
                pos_pts = self._sample_mask_points(top_coords, n=5)
            else:
                pos_pts = np.array([[cx, max(0, y1 - int(bh * 0.3))]], dtype=np.float32)

            # Negative points: 얼굴 전체에 격자 분포 (9점) + 목/배경
            neg_pts = np.array([
                # 얼굴 상단부
                [x1 + int(bw * 0.25), y1 + int(bh * 0.15)],  # 왼쪽 이마
                [cx,                   y1 + int(bh * 0.15)],  # 중앙 이마
                [x2 - int(bw * 0.25), y1 + int(bh * 0.15)],  # 오른쪽 이마
                # 얼굴 중앙부 (눈/코)
                [x1 + int(bw * 0.25), cy],                    # 왼쪽 뺨
                [cx,                   cy],                    # 얼굴 중심
                [x2 - int(bw * 0.25), cy],                    # 오른쪽 뺨
                # 얼굴 하단부 (입/턱)
                [x1 + int(bw * 0.25), y2 - int(bh * 0.15)],  # 왼쪽 아래
                [cx,                   y2 - int(bh * 0.15)],  # 턱
                [x2 - int(bw * 0.25), y2 - int(bh * 0.15)],  # 오른쪽 아래
                # 목
                [cx,                   y2 + int(bh * 0.2)],   # 목 중앙
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

                # multimask: BiSeNet face_region과 overlap 가장 적은 마스크 선택
                bisenet_dilated = self._dilate_mask(base_mask)

                # face 영역을 bbox 기반으로 빠르게 근사
                face_approx = np.zeros((H, W), dtype=np.float32)
                fy1 = max(0, int(y1 + bh * 0.1))
                fy2 = min(H, int(y2 - bh * 0.05))
                fx1 = max(0, int(x1 + bw * 0.05))
                fx2 = min(W, int(x2 - bw * 0.05))
                face_approx[fy1:fy2, fx1:fx2] = 1.0

                best_mask = None
                best_score = -1.0
                for m in masks_np:
                    m_f = (m > 0.5).astype(np.float32)
                    if m_f.shape != (H, W):
                        m_f = cv2.resize(m_f, (W, H), interpolation=cv2.INTER_LINEAR)
                        m_f = (m_f > 0.5).astype(np.float32)
                    # score = hair coverage - face penalty
                    hair_hit  = (m_f * bisenet_dilated).sum()
                    face_hit  = (m_f * face_approx).sum()
                    score = hair_hit - face_hit * 2.0   # face overlap에 2배 패널티
                    if score > best_score:
                        best_score = score
                        best_mask = m_f

                refined_np = best_mask
                # BiSeNet hair 영역 교집합으로 얼굴 내부 완전 제거
                refined_np = np.clip(refined_np * bisenet_dilated, 0.0, 1.0)
                if refined_np.sum() < 300:
                    logger.warning("[SDPipeline] SAM2∩BiSeNet 결과가 너무 작아 BiSeNet으로 폴백")
                    return self._dilate_mask(base_mask), "bisenet"

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

        # ── Canny edge (hair 영역은 약화 → SD가 자유롭게 생성)
        gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
        hair_soft = cv2.GaussianBlur(msk_canvas, (0, 0), sigmaX=7)
        canny_f = canny.astype(np.float32) * (1.0 - hair_soft * 0.8)
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
        """IP-Adapter용 얼굴 crop (padding + 224×224 resize)"""
        x1, y1, x2, y2 = face_bbox
        W, H = img_pil.size
        bw, bh = x2 - x1, y2 - y1
        px = int(bw * self.config.face_crop_padding)
        py = int(bh * self.config.face_crop_padding)
        crop = img_pil.crop((
            max(0, x1 - px), max(0, y1 - py),
            min(W, x2 + px), min(H, y2 + py),
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
            # 턱 바로 아래 (얼굴 높이의 +5%)
            cutoff_y = int(y2 + face_h * 0.05)
        else:  # medium
            # 어깨 위 (얼굴 높이의 +60%)
            cutoff_y = int(y2 + face_h * 0.60)
        cutoff_y = max(0, min(cutoff_y, H - 1))

        # 기존 머리카락의 최하단 행
        hair_rows = np.any(hair_mask > 0.5, axis=1)
        if not np.any(hair_rows):
            return hair_mask
        lowest_hair_y = int(np.max(np.where(hair_rows)))

        if lowest_hair_y <= cutoff_y:
            # 이미 요청한 길이보다 짧음 → 확장 불필요
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
                # 헤어 없는 행: 위에서 추정한 헤어 열 범위로만 채움 (얼굴/목 제외)
                c_min, c_max = default_c_min, default_c_max
            expanded[row, max(0, c_min):min(W, c_max + 1)] = 1.0

        return expanded

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
        parts = []
        if hairstyle_text:
            parts.append(hairstyle_text.strip())
        if color_text:
            parts.append(f"{color_text.strip()} hair color")
        style = ", ".join(parts) if parts else "natural hairstyle"

        # ── 길이별 positive/negative 보강 ────────────────────────────────────
        if hair_length == "short":
            pos_suffix = (
                ", short hairstyle, hair length above shoulders, "
                "visible neck, bare neck, short hair ends at chin or above"
            )
            neg_prefix = "long hair, long flowing hair, hair below shoulders, wavy long hair, "
            guidance = 9.5   # 프롬프트 추종 강화
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

        positive = (
            f"professional portrait photo of a person with {style}{pos_suffix}, "
            "photorealistic, high quality, natural lighting, 8k, "
            "studio photography, sharp focus, beautiful hair"
        )
        negative = neg_prefix + _NEGATIVE_BASE

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
    ) -> List[Image.Image]:
        """
        모든 seed를 단일 배치 forward pass로 생성 (순차 대비 ~절반 시간).

        diffusers는 generator를 리스트로 받으면 num_images_per_prompt 개의
        이미지를 각자 다른 seed로 한 번의 파이프라인 실행에 처리함.
        """
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
                controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
                num_images_per_prompt=n,
                generator=generators,
                strength=1.0,
            )

        logger.info(f"[SDPipeline] 배치 생성 완료 → {len(out.images)}장")
        return out.images

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
    ) -> np.ndarray:
        """
        SD 생성 이미지를 원본에 합성.
        - hair mask 영역: SD 생성 결과
        - 그 외: 원본 (얼굴/배경 유지)
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

        # alpha 블렌딩 (경계 gaussian smoothing)
        alpha = cv2.GaussianBlur(hair_mask, (0, 0), sigmaX=4.0, sigmaY=4.0)
        alpha = np.clip(alpha, 0.0, 1.0)[..., np.newaxis]   # H×W×1

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
        self._mp_face = None
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
