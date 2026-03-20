# MirrAI SD Inpainting Runtime

이 저장소의 현재 메인 경로는 `Stable Diffusion inpainting` 기반 헤어 변환 런타임입니다.  
지금 RunPod 서버리스에 올려서 테스트하고 있는 엔트리포인트는 `handler_sd.py`이고, 핵심 추론 로직은 `pipeline_sd_inpainting.py`에 있습니다.

기존 `pipeline_optimized.py` / `runpod_handler.py` 경로는 여전히 저장소에 남아 있지만, 현재 주력 생성 경로는 아닙니다. 그쪽은 추천/트렌드 분석 실험용 레거시 경로에 가깝고, 현재 SD 테스트 플로우와는 분리되어 있습니다.

## 현재 메인 구성

- `handler_sd.py`
  RunPod Serverless handler. 입력 파싱, cold start 모델 준비, 결과 직렬화를 담당합니다.
- `pipeline_sd_inpainting.py`
  현재 메인 헤어 생성 파이프라인입니다. 얼굴 보호, 헤어 마스크 정제, SD inpainting, 후처리, top-k ranking까지 담당합니다.
- `runtime_download.py`
  cold start 시 Hugging Face 모델과 `seg.pth`를 준비합니다.
- `test_runpod.py`
  배포된 RunPod endpoint에 요청을 보내고 결과 이미지를 저장하는 테스트 클라이언트입니다.
- `scripts/update_runpod_serverless.py`
  RunPod endpoint-bound template 이미지를 새 버전으로 교체할 때 사용합니다.
- `Dockerfile.sd.base`
  SD 런타임 베이스 이미지입니다.
- `Dockerfile.sd.app`
  코드만 얹는 앱 이미지입니다. 현재 `handler_sd.py` 기반 서버리스 워커를 띄웁니다.
- `entrypoint_sd.sh`
  앱 컨테이너 시작 스크립트입니다.

## 현재 모델 스택

- 얼굴 검출 / 랜드마크
  `MediaPipe FaceDetection`, `MediaPipe FaceMesh`
- 헤어 / 얼굴 / 옷 파싱
  `SegFace / BiSeNet` 계열 파서
  로컬 체크포인트: `pretrained_models/seg.pth`
- 헤어 마스크 정제
  `SAM2`
  로컬 체크포인트: `pretrained_models/sam2.pt`
- 생성 모델
  `runwayml/stable-diffusion-inpainting`
- 구조 가이드
  `lllyasviel/control_v11p_sd15_canny`
- 얼굴 아이덴티티 가이드
  `h94/IP-Adapter`
  weight: `ip-adapter-plus-face_sd15.bin`
- 대형 영역 배경 / 잔여물 보정
  `LaMa` 기반 fill
  필요 시 `bg_fill_mode=sd`로 대체 가능

## 현재 처리 프로세스

현재 SD 경로는 "추천을 먼저 하고 생성"하는 구조가 아니라, 이미 정해진 `hairstyle_text`와 `color_text`를 입력받아 바로 생성하는 구조입니다.

1. 입력 이미지와 `hairstyle_text`, `color_text`를 받습니다.
2. MediaPipe로 얼굴 bbox와 랜드마크를 검출합니다.
3. SegFace로 hair / face / cloth 영역을 파싱합니다.
4. face / cloth 보호 마스크를 만들고 생성 대상 hair mask를 정리합니다.
5. SAM2로 hair mask 경계를 보정합니다.
6. short / medium 변환일 때는 하단 긴머리 잔여물을 먼저 pre-clean 합니다.
7. Canny edge를 만들어 ControlNet conditioning에 사용합니다.
8. 얼굴 crop을 만들어 IP-Adapter conditioning에 사용합니다.
9. SD 1.5 inpainting으로 top-k seed 결과를 생성합니다.
10. 후처리, 합성, ranking을 거쳐 최종 결과를 반환합니다.

현재 ranking 결과에는 아래 점수가 포함됩니다.

- `clip_score`
- `color_score`
- `silhouette_score`
- `rank_score`

## 입력 스키마

현재 메인 handler(`handler_sd.py`)는 아래 형태를 받습니다.

```json
{
  "input": {
    "image": "<base64 or URL>",
    "hairstyle_text": "short chin-length bob cut, hush cut",
    "color_text": "ash beige",
    "top_k": 3,
    "bg_fill_mode": "lama",
    "seed": 1037080612,
    "return_base64": true,
    "return_intermediates": false
  }
}
```

지원 입력 키:

- `image`, `image_base64`, `image_url`, `image_path` 중 하나
- `hairstyle_text`
- `color_text`
- `top_k` (`1~5`)
- `bg_fill_mode` (`lama` 또는 `sd`)
- `seed` (선택)
- `return_base64` (선택)
- `return_intermediates` (선택)
- `cleanup_params` (ABI 호환용, 현재는 실질적으로 사용하지 않음)

주의:

- 현재 이 경로는 취향 추천, 트렌드 추천, 얼굴 분석 기반 스타일 추천을 직접 수행하지 않습니다.
- 그런 추천형 플로우는 `runpod_handler.py` + `pipeline_optimized.py` 쪽에 별도로 남아 있습니다.

## 출력 스키마

대표 응답 구조는 아래와 같습니다.

```json
{
  "results": [
    {
      "rank": 0,
      "seed": 1037080612,
      "clip_score": 0.615,
      "color_score": 0.201,
      "silhouette_score": 0.711,
      "rank_score": 0.572,
      "mask_used": "sam2",
      "image_base64": "..."
    }
  ],
  "elapsed_seconds": 64.2,
  "intermediates": {},
  "intermediate_data": {}
}
```

`return_intermediates=true`인 경우 rank 0 기준 중간 디버그 이미지와 메타데이터가 같이 반환됩니다.

## 설치 및 준비

현재 SD 런타임 기준 의존성 설치:

```bash
pip install -r requirements-sd.txt
```

권장 환경변수:

```bash
HF_TOKEN=hf_xxx
RUNPOD_API_KEY=rpa_xxx
RUNPOD_ENDPOINT_ID=your_endpoint_id
```

모델 준비:

- `pretrained_models/sam2.pt`는 로컬에 있어야 합니다.
- `pretrained_models/seg.pth`는 아래 중 하나로 준비됩니다.
  - 파일을 직접 마운트
  - `SEG_PTH_URL` 설정
  - `GITHUB_TOKEN` + 기본 GitHub 다운로드 경로 사용
- Hugging Face 기반 모델은 `runtime_download.py`가 cold start 시 캐시합니다.

## RunPod 테스트

배포된 endpoint를 기준으로 가장 많이 쓰는 테스트 명령은 아래입니다.

```bash
python test_runpod.py \
  --image images/test_2.jpg \
  --hairstyle "short chin-length bob cut, hush cut" \
  --color "ash beige"
```

URL 입력:

```bash
python test_runpod.py \
  --image-url https://example.com/photo.jpg \
  --hairstyle "bob cut" \
  --color "black"
```

헬스체크:

```bash
python test_runpod.py --health-check
```

중간 산출물까지 받고 싶으면 `test_runpod.py` 쪽 옵션을 사용해 `return_intermediates`를 켜서 확인하면 됩니다.

## 서버리스 워커 실행

현재 앱 이미지는 `handler_sd.py`를 직접 엔트리포인트로 사용합니다.

```bash
python handler_sd.py
```

이 명령은 일반 HTTP 서버를 띄우는 것이 아니라 `runpod.serverless.start(...)`를 호출해 RunPod worker 프로세스로 동작합니다.

## Docker / 배포 흐름

현재 배포는 보통 아래 흐름을 따릅니다.

1. `Dockerfile.sd.base`로 베이스 이미지를 준비합니다.
2. `Dockerfile.sd.app`으로 코드 전용 앱 이미지를 빌드합니다.
3. 새 이미지 태그를 RunPod template에 반영합니다.
4. endpoint worker rollout 완료 후 `test_runpod.py`로 검증합니다.

예시:

```bash
docker build -f Dockerfile.sd.base -t byoungj/sd:base-latest .

docker build \
  -f Dockerfile.sd.app \
  --build-arg BASE_IMAGE=byoungj/sd:base-latest \
  -t byoungj/sd:v129 .

python scripts/update_runpod_serverless.py \
  --image byoungj/sd:v129 \
  --wait
```

`entrypoint_sd.sh`는 기본적으로 이미지에 포함된 코드를 그대로 사용합니다.  
`ENABLE_STARTUP_GIT_PULL=1`일 때만 컨테이너 시작 시 git pull을 시도합니다.

## 레거시 / 보조 경로

아래 파일들은 아직 저장소에 남아 있지만, 현재 주력 SD 서버리스 테스트 경로와는 다릅니다.

- `runpod_handler.py`
- `pipeline_optimized.py`

이 경로는 추천 / 트렌드 / 분석 중심 플로우를 포함한 예전 런타임입니다.  
현재 `handler_sd.py` 기반 테스트 결과를 해석할 때는 이 경로의 입력 스키마와 혼동하지 않는 것이 중요합니다.
