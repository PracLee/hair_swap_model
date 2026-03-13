"""
마스킹 디버그 스크립트
SAM2 단독 vs BiSeNet 단독 vs 현재 파이프라인 마스크를 나란히 저장
"""
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from pipeline_sd_inpainting import MirrAISDPipeline, SDInpaintConfig, HAIR_CLASS_IDX, BISENET_CLASSES, FACE_CLASS_IDXS

IMAGE_PATH  = "images/1231.jpg"
OUTPUT_DIR  = PROJECT_ROOT

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[debug] device={device}")

    config = SDInpaintConfig()
    pipeline = MirrAISDPipeline(config=config)

    # BiSeNet + SAM2만 로드 (SD는 불필요)
    pipeline._load_bisenet()
    pipeline._load_sam2()
    pipeline._load_mediapipe()

    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # ── Step 1: 얼굴 검출
    face_bbox = pipeline._detect_face(img_rgb)
    print(f"[debug] face_bbox={face_bbox}")

    # ── Step 2: BiSeNet
    hair_mask_bisenet, face_region_mask = pipeline._bisenet_hair_mask(img_rgb)
    print(f"[debug] BiSeNet hair pixels={hair_mask_bisenet.sum():.0f}")

    # ── Step 3: SAM2 (raw, 교집합 없음)
    # SAM2 내부 코드를 직접 실행해서 raw 결과 저장
    try:
        predictor = pipeline._sam2_factory()
        x1, y1, x2, y2 = face_bbox
        bw, bh = x2 - x1, y2 - y1
        sam_bbox = np.array([
            max(0, x1 - int(bw * 0.15)),
            max(0, y1 - int(bh * 0.4)),
            min(W - 1, x2 + int(bw * 0.15)),
            min(H - 1, y2 + int(bh * 0.08)),
        ], dtype=np.float32)

        hair_coords = np.argwhere(hair_mask_bisenet > 0.5)
        pos_pts = pipeline._sample_mask_points(hair_coords, n=3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        neg_pts = np.array([
            [cx, cy],
            [x1 + int(bw * 0.1), cy],
            [x2 - int(bw * 0.1), cy],
        ], dtype=np.float32)
        point_coords = np.concatenate([pos_pts, neg_pts], axis=0)
        point_labels = np.concatenate([
            np.ones(len(pos_pts), dtype=np.int32),
            np.zeros(len(neg_pts), dtype=np.int32),
        ])

        predictor.set_image(img_rgb)
        prediction = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=sam_bbox[None, :],
            multimask_output=False,
        )
        masks = prediction[0] if isinstance(prediction, (tuple, list)) else prediction.get("masks")
        while isinstance(masks, (tuple, list)):
            masks = masks[0]
        masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
        sam2_raw = (masks_np[0] if masks_np.ndim == 3 else masks_np).astype(np.float32)
        sam2_raw = (sam2_raw > 0.5).astype(np.float32)
        if sam2_raw.shape != (H, W):
            sam2_raw = cv2.resize(sam2_raw, (W, H), interpolation=cv2.INTER_LINEAR)
            sam2_raw = (sam2_raw > 0.5).astype(np.float32)
        print(f"[debug] SAM2 raw pixels={sam2_raw.sum():.0f}")
    except Exception as e:
        print(f"[debug] SAM2 실패: {e}")
        sam2_raw = hair_mask_bisenet.copy()

    # ── 현재 파이프라인 마스크 (SAM2 ∩ BiSeNet - face)
    bisenet_dilated = pipeline._dilate_mask(hair_mask_bisenet)
    pipeline_mask = np.clip(sam2_raw * bisenet_dilated, 0.0, 1.0)
    pipeline_mask = np.clip(pipeline_mask - face_region_mask, 0.0, 1.0)

    # ── 저장
    def to_uint8(m):
        return (np.clip(m, 0, 1) * 255).astype(np.uint8)

    cv2.imwrite(str(OUTPUT_DIR / "debug_mask_bisenet.jpg"),    to_uint8(hair_mask_bisenet))
    cv2.imwrite(str(OUTPUT_DIR / "debug_mask_face_region.jpg"), to_uint8(face_region_mask))
    cv2.imwrite(str(OUTPUT_DIR / "debug_mask_sam2_raw.jpg"),   to_uint8(sam2_raw))
    cv2.imwrite(str(OUTPUT_DIR / "debug_mask_pipeline.jpg"),   to_uint8(pipeline_mask))

    # 원본 위에 마스크 오버레이 (빨간색)
    def overlay(img_rgb, mask, color=(255, 0, 0), alpha=0.5):
        overlay_img = img_rgb.copy()
        overlay_img[mask > 0.5] = (
            overlay_img[mask > 0.5] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)
        return cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(OUTPUT_DIR / "debug_overlay_bisenet.jpg"),  overlay(img_rgb, hair_mask_bisenet))
    cv2.imwrite(str(OUTPUT_DIR / "debug_overlay_sam2_raw.jpg"), overlay(img_rgb, sam2_raw))
    cv2.imwrite(str(OUTPUT_DIR / "debug_overlay_pipeline.jpg"), overlay(img_rgb, pipeline_mask))

    print("\n[debug] 저장 완료:")
    print("  debug_mask_bisenet.jpg    → BiSeNet 단독")
    print("  debug_mask_sam2_raw.jpg   → SAM2 단독 (raw)")
    print("  debug_mask_pipeline.jpg   → 현재 파이프라인 (SAM2∩BiSeNet - face)")
    print("  debug_overlay_*.jpg       → 원본 위에 오버레이")

if __name__ == "__main__":
    main()
