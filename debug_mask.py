import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_sd_inpainting import MirrAISDPipeline, SDInpaintConfig

IMAGE_PATH = str(PROJECT_ROOT / "images" / "12345.png")
OUTPUT_DIR = PROJECT_ROOT


def draw_points(img_bgr, pos_pts, neg_pts, sam_bbox):
    """positive(초록), negative(빨강) 포인트 + SAM2 bbox 시각화"""
    vis = img_bgr.copy()
    x1, y1, x2, y2 = [int(v) for v in sam_bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 165, 0), 2)  # 주황 박스

    for x, y in pos_pts:
        cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0), -1)   # 초록 = positive
        cv2.circle(vis, (int(x), int(y)), 8, (255,255,255), 2)

    for x, y in neg_pts:
        cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)   # 빨강 = negative
        cv2.circle(vis, (int(x), int(y)), 8, (255,255,255), 2)

    return vis


def main():
    parser = argparse.ArgumentParser(description="SAM2/BiSeNet mask debug")
    parser.add_argument(
        "--image",
        default=IMAGE_PATH,
        help="입력 이미지 경로 (기본: images/12345.png)",
    )
    args = parser.parse_args()

    pipeline = MirrAISDPipeline(config=SDInpaintConfig(device="cpu", dtype="bfloat16"))
    pipeline._load_bisenet()
    pipeline._load_mediapipe()

    img_bgr = cv2.imread(str(args.image))
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # ── BiSeNet hair + face mask
    hair_mask, face_mask = pipeline._bisenet_hair_mask(img_rgb)
    print(f"BiSeNet hair pixels: {hair_mask.sum():.0f}")
    print(f"BiSeNet face pixels: {face_mask.sum():.0f}")

    face_bbox = pipeline._detect_face(img_rgb)
    if face_bbox is None:
        raise RuntimeError("얼굴 bbox를 검출하지 못했습니다.")
    x1, y1, x2, y2 = face_bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    print(f"face_bbox: {face_bbox}")

    # ── SAM2 bbox (새 로직)
    sam_bbox = np.array([
        max(0,     x1 - int(bw * 0.5)),
        max(0,     y1 - int(bh * 0.6)),
        min(W - 1, x2 + int(bw * 0.5)),
        min(H - 1, y2 + int(bh * 0.15)),
    ], dtype=np.float32)

    # ── Positive points (새 로직 - face bbox 기반)
    hair_top_y = max(5, y1 - int(bh * 0.25))
    side_y     = max(5, y1 - int(bh * 0.05))
    pos_pts = np.array([
        [cx,                   hair_top_y],
        [cx - int(bw * 0.25),  hair_top_y],
        [cx + int(bw * 0.25),  hair_top_y],
        [x1 - int(bw * 0.05),  side_y],
        [x2 + int(bw * 0.05),  side_y],
    ], dtype=np.float32)
    pos_pts[:, 0] = np.clip(pos_pts[:, 0], 0, W - 1)
    pos_pts[:, 1] = np.clip(pos_pts[:, 1], 0, H - 1)

    # BiSeNet hair 보조 힌트
    hair_coords = np.argwhere(hair_mask > 0.5)
    if len(hair_coords) > 50:
        extra = pipeline._sample_mask_points(hair_coords, n=3)
        pos_pts = np.concatenate([pos_pts, extra], axis=0)

    # ── Negative points (얼굴 격자 9점 + 목)
    neg_pts = np.array([
        [x1 + int(bw * 0.25), y1 + int(bh * 0.25)],
        [cx,                   y1 + int(bh * 0.25)],
        [x2 - int(bw * 0.25), y1 + int(bh * 0.25)],
        [x1 + int(bw * 0.25), cy],
        [cx,                   cy],
        [x2 - int(bw * 0.25), cy],
        [x1 + int(bw * 0.25), y2 - int(bh * 0.15)],
        [cx,                   y2 - int(bh * 0.15)],
        [x2 - int(bw * 0.25), y2 - int(bh * 0.15)],
        [cx,                   y2 + int(bh * 0.2)],
    ], dtype=np.float32)
    neg_pts[:, 0] = np.clip(neg_pts[:, 0], 0, W - 1)
    neg_pts[:, 1] = np.clip(neg_pts[:, 1], 0, H - 1)

    # ── 시각화 저장
    vis = draw_points(img_bgr, pos_pts, neg_pts, sam_bbox)
    cv2.imwrite(str(OUTPUT_DIR / "debug_sam2_points.jpg"), vis)

    cv2.imwrite(str(OUTPUT_DIR / "debug_bisenet_hair.jpg"),
                (hair_mask * 255).astype(np.uint8))
    cv2.imwrite(str(OUTPUT_DIR / "debug_bisenet_face.jpg"),
                (face_mask * 255).astype(np.uint8))

    # BiSeNet face mask 제거 후 hair
    hair_clean = np.clip(hair_mask - face_mask, 0.0, 1.0)
    cv2.imwrite(str(OUTPUT_DIR / "debug_bisenet_hair_clean.jpg"),
                (hair_clean * 255).astype(np.uint8))

    print("\n저장 완료:")
    print("  debug_sam2_points.jpg      → SAM2 포인트/bbox 위치 확인 (초록=pos, 빨강=neg)")
    print("  debug_bisenet_hair.jpg     → BiSeNet hair mask")
    print("  debug_bisenet_face.jpg     → BiSeNet face mask")
    print("  debug_bisenet_hair_clean.jpg → hair - face 제거")


if __name__ == "__main__":
    main()
