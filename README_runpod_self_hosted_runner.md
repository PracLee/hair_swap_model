# RunPod Self-Hosted Runner 설정 가이드

이 문서는 GitHub Actions를 `GitHub hosted runner` 대신 `RunPod self-hosted runner`로 실행하기 위한 설정입니다.

## 1) 이미 반영된 저장소 변경사항

워크플로우 러너 타깃이 아래처럼 변경되어 있습니다.

- [build-sd-base.yml](/Users/leebyoungjae/Desktop/Workspaces/HearCLIP_workspace/workspace/.github/workflows/build-sd-base.yml)
- [build-sd-app.yml](/Users/leebyoungjae/Desktop/Workspaces/HearCLIP_workspace/workspace/.github/workflows/build-sd-app.yml)

```yaml
runs-on: [self-hosted, linux, x64, runpod, sd-builder]
```

즉, GitHub에 등록된 self-hosted runner가 위 라벨을 가지고 있어야 잡이 실행됩니다.

## 2) RunPod에서 직접 설정해야 하는 항목 (필수)

### A. RunPod Pod 준비

- 타입: 일반 Pod (Serverless Endpoint 아님)
- 권장 디스크: 최소 80GB 이상 (Docker layer/cache 용도)
- 네트워크: GitHub, Docker Hub outbound 가능해야 함

### B. GitHub Personal Access Token 준비

`GITHUB_RUNNER_PAT` 용도로 토큰을 생성하세요.

- Repository 접근 가능한 PAT 필요
- private repo면 `repo` 권한 포함 필요
- runner 등록 API 호출 가능 권한 필요

### C. Pod 환경변수

아래 값을 Pod에 넣어주세요.

- `GITHUB_OWNER=PracLee`
- `GITHUB_REPO=hair_swap_model`
- `GITHUB_RUNNER_PAT=<your_pat>`
- `RUNNER_NAME=runpod-sd-builder-01` (원하는 이름)
- `RUNNER_LABELS=runpod,sd-builder,linux,x64`

### D. Runner 시작

Pod 쉘에서 실행:

```bash
cd /workspace
bash scripts/setup_runpod_runner.sh
```

정상 등록되면 GitHub 저장소의 `Settings > Actions > Runners` 에서 runner가 `Online`으로 보입니다.

## 3) GitHub Repository에서 확인할 항목

### Actions Secrets (기존 유지)

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

### 워크플로우 실행 방식

- 코드 push 또는 `Run workflow` 시 self-hosted runner가 잡을 가져감
- runner가 offline이면 workflow가 대기 상태로 남음

## 4) 운영 팁

- Runner Pod는 계속 켜두는 것이 안정적입니다.
- Runner를 내릴 때는 GitHub Runners 화면에서 `Offline`/`stale` 상태를 정리하세요.
- Docker cache가 쌓이면 주기적으로 정리하세요:

```bash
docker system prune -af
```

