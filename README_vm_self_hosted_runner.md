# VM Self-Hosted Runner 설정 가이드

GitHub Actions의 SD 이미지 빌드를 VM self-hosted runner에서 실행하는 방법입니다.

## 1) 현재 워크플로우 라벨

다음 두 워크플로우는 아래 라벨을 가진 runner를 찾습니다.

- [build-sd-base.yml](/Users/leebyoungjae/Desktop/Workspaces/HearCLIP_workspace/workspace/.github/workflows/build-sd-base.yml)
- [build-sd-app.yml](/Users/leebyoungjae/Desktop/Workspaces/HearCLIP_workspace/workspace/.github/workflows/build-sd-app.yml)

```yaml
runs-on: [self-hosted, linux, x64, vm, sd-builder]
```

## 2) 사용자 설정 항목 (필수)

### A. VM 준비

- Ubuntu 22.04+ 권장
- 디스크 100GB+ 권장 (Docker layer/cache)
- 아웃바운드 네트워크:
  - `github.com`
  - `api.github.com`
  - `registry-1.docker.io`

### B. GitHub PAT 준비

runner 등록용 PAT 생성:

- private repo면 `repo` 권한 포함
- 토큰은 유출 시 즉시 폐기/재발급

### C. VM에서 환경변수 설정

```bash
export GITHUB_OWNER="PracLee"
export GITHUB_REPO="hair_swap_model"
export GITHUB_RUNNER_PAT="<YOUR_PAT>"
export RUNNER_NAME="vm-sd-builder-01"
export RUNNER_LABELS="vm,sd-builder,linux,x64"
```

### D. runner 설치/실행

```bash
cd /path/to/hair_swap_model
bash scripts/setup_vm_runner.sh
```

정상 등록되면 GitHub 저장소의 `Settings > Actions > Runners` 에서 Online 상태를 확인할 수 있습니다.

## 3) GitHub Repository Secrets

다음 시크릿이 필요합니다.

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

## 4) 운영 체크포인트

- 러너가 offline이면 workflow는 대기합니다.
- 디스크가 차면 빌드 실패하므로 주기적으로 정리하세요.

```bash
docker system prune -af
```

