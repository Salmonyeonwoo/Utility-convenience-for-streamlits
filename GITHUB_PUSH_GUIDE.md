# GitHub에 푸시하는 방법

## 방법 1: GitHub Desktop 사용 (GUI)

1. GitHub Desktop 설치: https://desktop.github.com/
2. GitHub Desktop 실행
3. File → Add Local Repository
4. 경로 선택: `C:\Users\Admin\Downloads\Updated_streamlit_app_files`
5. 커밋 메시지 입력 후 "Commit to main"
6. "Push origin" 클릭

## 방법 2: 명령줄 사용 (Personal Access Token 필요)

### 1. GitHub Personal Access Token 생성
1. GitHub.com 로그인
2. Settings → Developer settings → Personal access tokens → Tokens (classic)
3. "Generate new token (classic)" 클릭
4. Note: "Streamlit App Push" 입력
5. 권한 선택: `repo` 체크
6. "Generate token" 클릭
7. **토큰을 복사해두세요! (한 번만 표시됩니다)**

### 2. Git 푸시 실행
```bash
cd "C:\Users\Admin\Downloads\Updated_streamlit_app_files"
git push -u origin main
```

인증 창이 뜨면:
- Username: GitHub 사용자명 입력
- Password: **Personal Access Token** 입력 (비밀번호가 아님!)

## 방법 3: SSH 키 사용 (고급)

SSH 키를 설정하면 인증 없이 푸시할 수 있습니다.

1. SSH 키 생성:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. 공개 키를 GitHub에 등록:
   - GitHub.com → Settings → SSH and GPG keys → New SSH key
   - 공개 키 내용 복사하여 등록

3. 원격 저장소를 SSH로 변경:
```bash
git remote set-url origin git@github.com:Salmonyeonwoo/Utility-convenience-for-streamlits.git
git push -u origin main
```

## 현재 상태 확인

```bash
cd "C:\Users\Admin\Downloads\Updated_streamlit_app_files"
git status
git log --oneline -1
git remote -v
```

## 문제 해결

### "Authentication failed" 오류
- Personal Access Token을 사용했는지 확인
- 토큰이 만료되지 않았는지 확인

### "Permission denied" 오류
- 저장소에 대한 쓰기 권한이 있는지 확인
- 저장소 소유자 또는 Collaborator 권한이 필요합니다

### "Repository not found" 오류
- 저장소 URL이 정확한지 확인
- 저장소가 Private인 경우 토큰에 적절한 권한이 있는지 확인

---

## 🔧 CI/CD 설정 가이드 (GitHub Actions)

이 프로젝트는 GitHub Actions를 사용하여 자동화된 CI/CD 파이프라인을 구성합니다.

### 설정된 워크플로우

프로젝트에는 다음 3개의 GitHub Actions 워크플로우가 설정되어 있습니다:

1. **Python package** (`.github/workflows/python-package.yml`)
   - 여러 Python 버전(3.9, 3.10, 3.11, 3.12)에서 패키지 빌드 및 테스트
   - `pyproject.toml`을 사용하여 패키지 빌드
   - 의존성 설치 및 패키지 import 테스트

2. **Pylint** (`.github/workflows/pylint.yml`)
   - Python 코드 품질 검사
   - `.pylintrc` 설정 파일 사용
   - 모든 Python 파일에 대한 린팅 검사

3. **Python application** (`.github/workflows/python-application.yml`)
   - 여러 Python 버전에서 애플리케이션 테스트
   - Python 구문 검사
   - 핵심 모듈 import 테스트
   - 애플리케이션 구조 검증

### 로컬에서 테스트하기

GitHub에 푸시하기 전에 로컬에서 테스트할 수 있습니다:

```bash
cd "C:\Users\Admin\Downloads\Updated_streamlit_app_files"
python test_local_setup.py
```

이 스크립트는 다음을 확인합니다:
- Python 버전 확인
- 의존성 설치 확인
- Python 구문 검사
- Import 테스트
- 필수 파일 존재 확인
- GitHub Actions 워크플로우 파일 확인

### 필요한 설정 파일

CI/CD가 정상적으로 작동하려면 다음 파일들이 필요합니다:

1. **`pyproject.toml`**
   - Python 패키지 메타데이터 및 의존성 정의
   - 빌드 시스템 설정
   - 개발 도구 설정 (pylint, black, mypy)

2. **`.pylintrc`**
   - Pylint 코드 품질 검사 규칙
   - 프로젝트에 맞게 최적화된 설정

3. **`.github/workflows/*.yml`**
   - GitHub Actions 워크플로우 정의 파일들

### 워크플로우 실행 확인

1. GitHub 저장소로 이동
2. **Actions** 탭 클릭
3. 워크플로우 실행 상태 확인:
   - ✅ 초록색 체크: 성공
   - ⏳ 노란색 원: 진행 중
   - ❌ 빨간색 X: 실패

### 워크플로우 트리거

워크플로우는 다음 이벤트에서 자동으로 실행됩니다:
- `main` 또는 `master` 브랜치에 push
- `main` 또는 `master` 브랜치로의 Pull Request

### 수동으로 워크플로우 실행하기

1. GitHub 저장소의 **Actions** 탭으로 이동
2. 실행하려는 워크플로우 선택 (예: "Python package")
3. 오른쪽의 **"Run workflow"** 버튼 클릭
4. 브랜치 선택 후 **"Run workflow"** 클릭

### 문제 해결

#### 워크플로우가 실행되지 않는 경우
- `.github/workflows/` 디렉토리가 올바른 위치에 있는지 확인
- YAML 파일의 문법 오류 확인
- GitHub 저장소의 Actions가 활성화되어 있는지 확인

#### 워크플로우가 실패하는 경우
- **Python package** 실패:
  - `pyproject.toml` 파일 확인
  - `requirements.txt`의 의존성 확인
  - Python 버전 호환성 확인

- **Pylint** 실패:
  - `.pylintrc` 설정 확인
  - 코드 품질 문제 수정
  - `pylint` 설치 확인

- **Python application** 실패:
  - Python 구문 오류 확인
  - Import 경로 확인
  - 필수 파일 존재 확인

### 추가 정보

- GitHub Actions 문서: https://docs.github.com/en/actions
- Pylint 문서: https://pylint.readthedocs.io/
- Python Packaging 가이드: https://packaging.python.org/
































































