# HairCLIPv2 RunPod 패키지

최적화된 RunPod 실행 경로만 남긴 HairCLIPv2 런타임 패키지입니다.  
주요 진입점은 [`pipeline_optimized.py`](/workspace/pipeline_optimized.py) 와 [`runpod_handler.py`](/workspace/runpod_handler.py) 입니다.

## 포함된 구성

- `pipeline_optimized.py`: 추천, 정렬, 생성, 합성까지 포함한 CLI 실행 경로
- `runpod_handler.py`: RunPod serverless 진입점
- `download_weights.py`: 런타임 가중치 복구, 검증, 재다운로드
- `install_requirements.sh`: 의존성 설치 + 가중치 bootstrap
- `models/stylegan2`, `models/face_parsing`, `models/bald_proxy`: 실제 추론 코드
- `utils/sam3_runtime.py`, `utils/sam_masking.py`: SAM3 및 BiRefNet 마스킹 refinement

## 필수 모델 파일

아래 파일은 `pretrained_models/` 아래에 있어야 합니다.

- `ffhq.pt`
- `seg.pth`
- `bald_proxy.pt`
- `ffhq_PCA.npz`
- `sam3.pt` (SAM3를 쓸 때만 필요)

아래 파일은 `criteria/lpips/weights/v0.1/` 아래에 있어야 합니다.

- `vgg.pth`

## .env 설정

이제 `.env` 파일에 토큰을 넣어두면 됩니다.  

- `HF_TOKEN=hf_your_token_here`

## 설치 및 준비

기본 런타임 및 BiRefNet(정밀 마스킹) 설치:

```bash
# BiRefNet 구동에 필요한 패키지 설치
pip install transformers timm torchvision huggingface_hub
bash install_requirements.sh
```

## 실행 명령어 정리

### 1. 스타일 추천만 실행
이미지를 분석하여 어울리는 헤어스타일 목록과 이유를 출력합니다.
```bash
python pipeline_optimized.py --image images/1231.jpg --text "trendy" --recommend-only
```

### 2. 스타일 생성 (기본)
추천된 스타일 중 상위 1개를 생성합니다. (`--trend-limit`으로 개수 조절)
```bash
python pipeline_optimized.py \
  --image images/1231.jpg \
  --text "trendy" \
  --trend-limit 1 \
  --output-dir output/
```

### 3. 정밀 마스킹 (BiRefNet) 적용 생성
현재 **BiRefNet**이 기본 정밀 마스킹 모델로 설정되어 있습니다. `transformers` 패키지가 설치되어 있으면 자동으로 활성화됩니다.
- **특징**: 잔머리, 머리카락 끝부분 등 육안 식별이 어려운 경계선을 매우 정밀하게 추출합니다.
- **작동**: BiRefNet(인물 전체 윤곽) ∩ BiSeNet(헤어 라벨) 교집합 방식을 사용하여 몸/얼굴을 제외한 정밀한 머리카락 영역만 추출합니다.

```bash
# 별도 옵션 없이도 정밀 마스킹이 적용됩니다. (JSON 결과에서 source: "birefnet" 확인)
python pipeline_optimized.py --image images/1231.jpg --text "chic style" --output-dir output/
```

### 4. 마스킹 단독 테스트 및 디버깅
모델별 마스킹 결과만 이미지 파일로 확인하고 싶을 때 사용합니다.

- **BiRefNet (인물 전체 윤곽)**:
  ```bash
  python test_birefnet.py
  # 결과: output/birefnet_mask.png (인물 실루엣)
  ```
- **BiSeNet (기본 헤어 파싱)**:
  ```bash
  python test_bisenet.py
  # 결과: output/bisenet_hair_mask.png (거친 헤어 영역)
  ```

## 마스킹 Refinement 모델 안내

현재 시스템은 두 가지 정밀 마스킹 엔진을 지원합니다:

1. **BiRefNet (권장)**: 
   - **장점**: 가장 정밀한 경계선 추출, 잔머리 보존 우수.
   - **상태**: 현재 서버 환경에서 가장 안정적으로 동작합니다.
   - **확인**: 결과 JSON에서 `"masking": {"source": "birefnet"}` 확인.

2. **SAM3 (Segment Anything 3)**:
   - **주의**: 특정 CUDA 환경에서 메모리 충돌(Segmentation Fault)이 발생할 수 있습니다. 
   - **활성화**: `--enable-sam3` 옵션 사용 시 동작 시도.

## RunPod 실행

```bash
python runpod_handler.py
```

## 현재 확인된 사항 및 팁

- **마스크 확인**: 스타일 생성 시 `output/` 폴더에 `{스타일명}_mask.png` 파일이 함께 저장되어, 실제 적용된 영역을 확인할 수 있습니다.
- **속도 최적화**: 동일 이미지를 반복 수정 시 `--preset realtime` 사용을 권장합니다.
- **가중치**: `ffhq_PCA.npz` 등이 누락되면 오류가 발생하므로 `download_weights.py`로 확인하십시오.
