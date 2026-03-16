#!/usr/bin/env python3
"""
RunPod Serverless 테스트 스크립트 — SD Inpainting 파이프라인
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

헤어스타일 생성 요청 → base64 응답 → JPG 저장

사용법:
    # .env에 RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID 설정 후
    python test_runpod.py --image /path/to/photo.jpg

    # 헤어스타일/색상 명시
    python test_runpod.py --image /path/to/photo.jpg \\
        --hairstyle "wolf cut, layered bangs" \\
        --color "ash brown"

    # URL로 이미지 전달
    python test_runpod.py --image-url https://example.com/photo.jpg \\
        --hairstyle "bob cut" --color "black"

    # 헬스체크만
    python test_runpod.py --health-check

    # 환경변수 직접 전달
    python test_runpod.py --api-key rpa_xxx --endpoint-id yyy --image photo.jpg
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
from urllib.parse import unquote, urlparse
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

RUNPOD_BASE_URL   = "https://api.runpod.ai/v2"
POLL_INTERVAL_SEC = 5
TIMEOUT_SEC       = 1200  # 20분: cold start 시 모델 다운로드(SD+ControlNet+IP-Adapter) 포함
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

INTERMEDIATE_KEY_ALIASES = {
    # legacy keys -> standardized keys
    "input_image": "pipeline_input_image",
    "hand_mask": "mediapipe_hand_mask",
    "face_region_mask": "segface_face_region_mask",
    "cloth_mask": "segface_cloth_mask",
    "hair_mask_after_face_protect": "pipeline_hair_mask_face_protected",
    "cloth_mask_dilated": "segface_cloth_mask_dilated",
    "hair_mask_after_cloth_protect": "pipeline_hair_mask_cloth_protected",
    "shoulder_protect_mask": "pipeline_shoulder_protect_mask",
    "short_removal_mask": "pipeline_short_removal_mask",
    "short_generation_mask": "pipeline_short_generation_mask",
    "background_cleaned_rgb": "cv2_background_cleaned_rgb",
    "sd_mask_512": "sd_inpaint_mask_512",
    "canny_512": "controlnet_canny_512",
    "generated_rank0_512": "sd_generated_rank0_512",
}


# ── RunPod API ─────────────────────────────────────────────────────────────────

def submit_job(endpoint_id: str, api_key: str, payload: dict) -> str:
    url = f"{RUNPOD_BASE_URL}/{endpoint_id}/run"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"input": payload},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    job_id = data["id"]
    print(f"[submit] job_id={job_id}  status={data['status']}")
    return job_id


def poll_job(endpoint_id: str, api_key: str, job_id: str) -> dict:
    url = f"{RUNPOD_BASE_URL}/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + TIMEOUT_SEC

    while time.time() < deadline:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "UNKNOWN")

        if status == "COMPLETED":
            print(f"\n[poll]   status={status}")
            print(f"[worker] workerId={data.get('workerId')}")

            # 대용량 응답 → output_url로 분리되는 경우
            output_url = data.get("output_url")
            if output_url:
                print(f"[debug]  output_url 감지 → 다운로드 중...")
                r2 = requests.get(output_url, timeout=120)
                r2.raise_for_status()
                return r2.json()

            output = data.get("output", {})
            if not output:
                print(f"[debug]  raw response (2000자): {json.dumps(data, ensure_ascii=False)[:2000]}")
            return output

        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            error = data.get("error") or data.get("output", {}).get("error", "")
            raise RuntimeError(f"Job {status}: {error}")

        elapsed = int(TIMEOUT_SEC - (deadline - time.time()))
        remaining = int(deadline - time.time())
        print(f"[poll]   status={status}  elapsed={elapsed}s  "
              f"(남은 대기: {remaining}s)", end="\r")
        time.sleep(POLL_INTERVAL_SEC)

    raise TimeoutError(f"Job did not complete within {TIMEOUT_SEC}s")


# ── 이미지 처리 ────────────────────────────────────────────────────────────────

def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def sanitize_filename_token(text: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("._-")
    token = token.replace(".", "_")
    token = re.sub(r"_+", "_", token).strip("_")
    return token.lower()


def infer_filename_prefix(image_path: Path | None, image_url: str | None) -> str:
    if image_path:
        return sanitize_filename_token(image_path.stem) or "input"

    if image_url:
        parsed = urlparse(image_url)
        name = Path(unquote(parsed.path)).name
        if name:
            return sanitize_filename_token(Path(name).stem) or "input"
    return "input"


def prefixed_name(prefix: str, base_name: str) -> str:
    clean_prefix = sanitize_filename_token(prefix)
    if not clean_prefix:
        return base_name
    return f"{clean_prefix}_{base_name}"


def save_results(output: dict, out_dir: Path, file_prefix: str) -> list[Path]:
    """결과 배열에서 image_base64 추출 → JPG 저장"""
    results = output.get("results", [])
    if not results:
        print(f"[warn] 응답에 results가 없습니다:")
        print(json.dumps(output, indent=2, ensure_ascii=False)[:1000])
        return []

    saved = []
    for item in results:
        rank       = item.get("rank", 0)
        seed       = item.get("seed", 0)
        clip_score = item.get("clip_score", 0.0)
        mask_used  = item.get("mask_used", "unknown")
        b64        = item.get("image_base64")

        if not b64:
            print(f"[warn] rank={rank}: image_base64 없음, 스킵")
            continue

        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        fname = prefixed_name(file_prefix, f"result_rank{rank}_seed{seed}_clip{clip_score:.3f}_{mask_used}.jpg")
        out_path = out_dir / fname
        img.save(out_path, format="JPEG", quality=95)

        # 디버그용 마스크 저장 (rank0만 — 모든 rank가 동일한 마스크 공유)
        if rank == 0:
            mask_b64 = item.get("mask_base64")
            if mask_b64:
                mask_bytes = base64.b64decode(mask_b64)
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("RGB")
                mask_fname = prefixed_name(file_prefix, f"mask_{mask_used}.jpg")
                mask_img.save(out_dir / mask_fname, format="JPEG", quality=95)
                print(f"[save]   mask        → {mask_fname}")

            overlay_b64 = item.get("mask_overlay_base64")
            if overlay_b64:
                ov_bytes = base64.b64decode(overlay_b64)
                ov_img = Image.open(io.BytesIO(ov_bytes)).convert("RGB")
                ov_fname = prefixed_name(file_prefix, f"mask_overlay_{mask_used}.jpg")
                ov_img.save(out_dir / ov_fname, format="JPEG", quality=95)
                print(f"[save]   mask_overlay → {ov_fname}")

        print(f"[save]   rank={rank}  seed={seed}  clip={clip_score:.3f}  "
              f"mask={mask_used}  → {out_path.name}")
        saved.append(out_path)

    return saved


def save_intermediates(output: dict, out_dir: Path, file_prefix: str) -> list[Path]:
    """top-level intermediates(base64 dict) 저장"""
    intermediates = output.get("intermediates") or {}
    if not isinstance(intermediates, dict) or not intermediates:
        return []

    saved: list[Path] = []
    result0 = (output.get("results") or [{}])[0] if isinstance(output.get("results"), list) else {}
    mask_used = str((result0 or {}).get("mask_used", "pipeline")).strip().lower() or "pipeline"
    for key, b64 in intermediates.items():
        if not b64:
            continue
        try:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"[warn] intermediate decode 실패: {key} ({e})")
            continue

        normalized_key = normalize_intermediate_key(key, mask_used=mask_used)
        out_path = out_dir / prefixed_name(file_prefix, f"intermediate_{normalized_key}.jpg")
        img.save(out_path, format="JPEG", quality=92)
        print(f"[save]   intermediate → {out_path.name}")
        saved.append(out_path)
    return saved


def save_intermediate_data(output: dict, out_dir: Path, file_prefix: str) -> list[Path]:
    """
    top-level intermediate_data(JSON) 저장.
    - intermediate_data.json
    - (있으면) mediapipe_face_mesh 랜드마크 CSV
    """
    payload = output.get("intermediate_data")
    if not isinstance(payload, dict) or not payload:
        return []

    saved: list[Path] = []

    json_path = out_dir / prefixed_name(file_prefix, "intermediate_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[save]   intermediate_data → {json_path.name}")
    saved.append(json_path)

    mesh = payload.get("mediapipe_face_mesh")
    if isinstance(mesh, dict):
        lms_norm = mesh.get("landmarks_norm")
        lms_px = mesh.get("landmarks_px")
        if isinstance(lms_norm, list) and isinstance(lms_px, list) and len(lms_norm) == len(lms_px):
            csv_path = out_dir / prefixed_name(file_prefix, "intermediate_mediapipe_face_mesh_landmarks.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "x_norm", "y_norm", "z_norm", "x_px", "y_px"])
                for idx, (n, p) in enumerate(zip(lms_norm, lms_px)):
                    if not isinstance(n, list) or len(n) < 3 or not isinstance(p, list) or len(p) < 2:
                        continue
                    writer.writerow([idx, n[0], n[1], n[2], p[0], p[1]])
            print(f"[save]   intermediate_data → {csv_path.name}")
            saved.append(csv_path)

    return saved


def normalize_intermediate_key(key: str, mask_used: str = "pipeline") -> str:
    norm = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(key)).strip("_").lower()
    norm = norm.replace(".", "_").replace("-", "_")
    norm = re.sub(r"_+", "_", norm).strip("_")
    if norm == "refined_hair_mask":
        source = re.sub(r"[^a-z0-9_]+", "_", (mask_used or "pipeline").lower()).strip("_")
        norm = f"{source}_refined_hair_mask"
    norm = INTERMEDIATE_KEY_ALIASES.get(norm, norm)
    if "_" not in norm:
        norm = f"pipeline_{norm}"
    return norm or "pipeline_unknown"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RunPod SD Inpainting 헤어스타일 테스트")

    # 인증
    parser.add_argument("--api-key",     default=os.environ.get("RUNPOD_API_KEY"),
                        help="RunPod API Key")
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"),
                        help="Serverless Endpoint ID")

    # 이미지 입력 (셋 중 하나)
    img_group = parser.add_mutually_exclusive_group()
    img_group.add_argument("--image",     type=Path, default=None,
                           help="로컬 이미지 경로 (base64로 전송)")
    img_group.add_argument("--image-url", default=None,
                           help="이미지 URL (컨테이너가 직접 다운로드)")

    # 헤어스타일 파라미터
    parser.add_argument("--hairstyle", default="wolf cut, layered bangs",
                        help="헤어스타일 텍스트 (기본: 'wolf cut, layered bangs')")
    parser.add_argument("--color",     default="ash brown",
                        help="헤어 컬러 텍스트 (기본: 'ash brown')")
    parser.add_argument("--top-k",     default=3, type=int,
                        help="결과 수 (기본: 3, 최대: 5)")
    parser.add_argument("--bg-fill",   default="cv2", choices=["cv2", "sd"],
                        help="단발 변환 시 배경 채우기 방법: cv2=기본(짧은길이 안정), sd=배경 품질 실험")

    # 기타
    parser.add_argument("--no-base64",    action="store_true",
                        help="이미지 base64 응답 비활성화 (메타정보만)")
    parser.add_argument("--return-intermediates", action="store_true",
                        help="중간 산출물(base64)을 함께 요청/저장")
    parser.add_argument("--health-check", action="store_true",
                        help="헬스체크만 실행 (이미지 불필요)")
    parser.add_argument("--output-dir",   type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="결과 이미지 저장 디렉토리 (기본: output/)")
    parser.add_argument("--filename-prefix", default=None,
                        help="저장 파일명 접두어 (기본: 입력 이미지 파일명)")
    parser.add_argument("--timeout",      type=int, default=TIMEOUT_SEC,
                        help=f"최대 대기 시간(초) (기본: {TIMEOUT_SEC}, cold start 시 넉넉히)")

    args = parser.parse_args()

    # ── 인증 검증 ──────────────────────────────────────────────────────────────
    if not args.api_key:
        sys.exit("❌  RUNPOD_API_KEY 환경변수 또는 --api-key 필요")
    if not args.endpoint_id:
        sys.exit("❌  RUNPOD_ENDPOINT_ID 환경변수 또는 --endpoint-id 필요")

    # ── 페이로드 구성 ──────────────────────────────────────────────────────────
    if args.health_check:
        payload = {"health_check": True}
    else:
        payload = {
            "hairstyle_text": args.hairstyle,
            "color_text":     args.color,
            "top_k":          args.top_k,
            "return_base64":  not args.no_base64,
            "return_intermediates": args.return_intermediates,
            "bg_fill_mode":   args.bg_fill,
        }

        if args.image_url:
            payload["image_url"] = args.image_url
        elif args.image:
            if not args.image.is_file():
                sys.exit(f"❌  이미지 파일 없음: {args.image}")
            payload["image"] = image_to_base64(args.image)
            print(f"[input]  이미지: {args.image.name}  ({args.image.stat().st_size // 1024} KB)")
        else:
            sys.exit("❌  --image 또는 --image-url 중 하나 필요 (--health-check 제외)")

    print(f"\n{'='*60}")
    print(f"Endpoint : {args.endpoint_id}")
    if not args.health_check:
        print(f"Hairstyle: {args.hairstyle}")
        print(f"Color    : {args.color}")
        print(f"Top-K    : {args.top_k}")
        print(f"BG Fill  : {args.bg_fill}")
        print(f"Intermed : {'on' if args.return_intermediates else 'off'}")
    print(f"{'='*60}\n")

    # ── 1. Job 제출 ────────────────────────────────────────────────────────────
    job_id = submit_job(args.endpoint_id, args.api_key, payload)

    # ── 2. 완료 대기 ───────────────────────────────────────────────────────────
    timeout = getattr(args, "timeout", TIMEOUT_SEC)
    if not args.health_check:
        print(f"[info]   cold start 시 모델 다운로드로 최대 {timeout//60}분 소요 가능")
    output = poll_job(args.endpoint_id, args.api_key, job_id)

    # ── 3. 에러 처리 ───────────────────────────────────────────────────────────
    if "error" in output:
        print(f"\n❌  서버 에러:\n{json.dumps(output['error'], indent=2, ensure_ascii=False)}")
        sys.exit(1)

    # ── 4. 헬스체크 결과 출력 ─────────────────────────────────────────────────
    if args.health_check:
        print("\n✅  헬스체크 결과:")
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    # ── 5. 결과 저장 ───────────────────────────────────────────────────────────
    # 처리 시간 출력
    elapsed = output.get("elapsed_seconds")
    if elapsed:
        print(f"\n[perf]   서버 처리 시간: {elapsed:.1f}s")

    file_prefix = args.filename_prefix or infer_filename_prefix(args.image, args.image_url)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = save_results(output, args.output_dir, file_prefix=file_prefix)
    saved_intermediates = save_intermediates(output, args.output_dir, file_prefix=file_prefix)
    saved_intermediate_data = save_intermediate_data(output, args.output_dir, file_prefix=file_prefix)

    if saved:
        print(f"\n✅  {len(saved)}개 이미지 저장 완료")
        for p in saved:
            print(f"   {p}")
        if saved_intermediates:
            print(f"\n✅  중간 산출물 {len(saved_intermediates)}개 저장 완료")
            for p in saved_intermediates:
                print(f"   {p}")
        if saved_intermediate_data:
            print(f"\n✅  분석 데이터 {len(saved_intermediate_data)}개 저장 완료")
            for p in saved_intermediate_data:
                print(f"   {p}")
    else:
        print("\n⚠️  저장된 이미지 없음")
        print(json.dumps(output, indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
