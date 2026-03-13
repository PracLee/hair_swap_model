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

# ── SegFace 설정 ───────────────────────────────────────────────────────────────
HAIR_CLASS_IDX   = 14
# 0: bg, 1: neck, 2: face, 3: cloth, 4: r_ear, 5: l_ear, 6: r_bro, 7: l_bro, 
# 8: r_eye, 9: l_eye, 10: nose, 11: inner_mouth, 12: lower_lip, 13: upper_lip
# 얼굴 내부 및 목/귀 클래스 포함
FACE_CLASS_IDXS  = frozenset([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

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
    mask: Optional[np.ndarray] = None       # H×W float32 디버그용 마스크
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)


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

        # ── Step 2: SegFace base hair mask + 얼굴 픽셀 마스크 ───────────────────
        hair_mask_base, face_region_mask = self._segface_hair_mask(img_rgb, face_bbox)

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

        # ── Step 3-c: SegFace 얼굴 픽셀 제거 (bbox 직사각형 대신 픽셀 단위 보정) ─
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
                face_bbox=face_bbox,
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
        segface.load_state_dict(ckpt)
        segface.to(self.device).eval()
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

    def _segface_hair_mask(self, img_rgb: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        SegFace로 머리카락 + 얼굴 영역 마스크 생성.
        얼굴 bbox를 기준으로 여유 있게 크롭한 뒤 512x512로 리사이즈하여 SegFace에 입력.
        이후 원본 해상도의 전체 영역으로 다시 복원하여 출력.
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
        
        if self.dtype == torch.float16:
            inp_t = inp_t.half()
        inp_t = inp_t.to(self.device)

        with torch.no_grad():
            DUMMY_LABELS = None
            DUMMY_DATASET = None
            logits = self._segface(inp_t, DUMMY_LABELS, DUMMY_DATASET)
            # logits: [1, 19, 512, 512]
            parsing = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        hair_512 = (parsing == HAIR_CLASS_IDX).astype(np.float32)
        face_512 = np.isin(parsing, list(FACE_CLASS_IDXS)).astype(np.float32)

        # 1. 원본 비율 (crop_w, crop_h) 해상도로 다시 리사이즈
        hair_crop = cv2.resize(hair_512, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        face_crop = cv2.resize(face_512, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. 패딩 부분 잘라내기
        hair_crop = hair_crop[:ch, :cw]
        face_crop = face_crop[:ch, :cw]
        
        # 3. 원본 HxW 해상도에 덮어쓰기
        hair_orig = np.zeros((H, W), dtype=np.float32)
        face_orig = np.zeros((H, W), dtype=np.float32)
        
        hair_orig[crop_y1:crop_y2, crop_x1:crop_x2] = hair_crop
        face_orig[crop_y1:crop_y2, crop_x1:crop_x2] = face_crop

        return (hair_orig > 0.5).astype(np.float32), (face_orig > 0.5).astype(np.float32)

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

                # multimask: 얼굴 overlap 가장 적고 hair 영역 많은 마스크 선택
                # hair_approx: 얼굴 위쪽 + 얼굴 옆/아래 측면 (긴 머리 포함)
                hair_approx = np.zeros((H, W), dtype=np.float32)
                # 정수리 영역
                hair_approx[max(0, y1 - int(bh*0.8)):min(H, y1 + int(bh*0.1)),
                            max(0, x1 - int(bw*0.5)):min(W, x2 + int(bw*0.5))] = 1.0
                # 얼굴 옆 + 아래쪽 측면 (긴 머리 흘러내리는 구간)
                hair_approx[y1:min(H, y2 + int(bh*1.2)),
                            max(0, x1 - int(bw*0.6)):max(0, x1 - int(bw*0.05))] = 1.0  # 왼쪽
                hair_approx[y1:min(H, y2 + int(bh*1.2)),
                            min(W, x2 + int(bw*0.05)):min(W, x2 + int(bw*0.6))] = 1.0  # 오른쪽

                # face 영역 근사 (페널티용)
                face_approx = np.zeros((H, W), dtype=np.float32)
                fy1 = max(0, int(y1 + bh * 0.1))
                fy2 = min(H, int(y2))
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
                    # 얼굴 위쪽 hair 영역 hit - 얼굴 overlap 2배 패널티
                    hair_hit = (m_f * hair_approx).sum()
                    face_hit = (m_f * face_approx).sum()
                    score = hair_hit - face_hit * 2.0
                    if score > best_score:
                        best_score = score
                        best_mask = m_f

                refined_np = best_mask
                # BiSeNet AND 교집합 제거 — BiSeNet이 hair를 일부만 감지하면 SAM2 결과도 잘려버림
                # 얼굴 제거는 run()의 face_region_mask 빼기(line 251)에서 처리
                if refined_np.sum() < 300:
                    logger.warning("[SDPipeline] SAM2 결과가 너무 작아 BiSeNet으로 폴백")
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
