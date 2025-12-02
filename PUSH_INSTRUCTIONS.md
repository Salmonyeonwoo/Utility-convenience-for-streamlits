# GitHub 푸시 명령줄 가이드

## 현재 상태
- ✅ Git 저장소 초기화 완료
- ✅ 원격 저장소 연결 완료: https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits.git
- ✅ 커밋 완료: "모듈화된 Streamlit 앱 구조로 리팩토링"

## 푸시 방법

### 방법 1: Personal Access Token 사용 (권장)

#### 1단계: GitHub에서 Personal Access Token 생성

1. **GitHub.com에 로그인**
2. **우측 상단 프로필 아이콘 클릭** → **Settings**
3. **왼쪽 메뉴에서 "Developer settings" 클릭**
4. **"Personal access tokens"** → **"Tokens (classic)"** 클릭
5. **"Generate new token"** → **"Generate new token (classic)"** 클릭
6. **Note 입력**: `Streamlit App Push` (원하는 이름)
7. **Expiration**: 원하는 기간 선택 (예: 90 days)
8. **권한 선택**: `repo` 체크박스 선택 (전체 repo 권한)
9. **"Generate token"** 클릭
10. **⚠️ 중요: 토큰을 복사해두세요!** (한 번만 표시됩니다)
    - 예: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

#### 2단계: Git 푸시 실행

명령 프롬프트에서 다음 명령어 실행:

```bash
cd "C:\Users\Admin\Downloads\Updated_streamlit_app_files"
git push -u origin main
```

**인증 창이 뜨면:**
- **Username**: GitHub 사용자명 입력 (예: `Salmonyeonwoo`)
- **Password**: **Personal Access Token 입력** (일반 비밀번호가 아님!)

### 방법 2: GitHub Desktop 사용

1. GitHub Desktop 실행
2. File → Add Local Repository
3. 경로 선택: `C:\Users\Admin\Downloads\Updated_streamlit_app_files`
4. 왼쪽에서 변경사항 확인
5. 하단에 커밋 메시지 입력
6. "Commit to main" 클릭
7. 상단 "Push origin" 클릭

### 방법 3: Git Credential Manager 사용 (Windows)

Windows에서 Git Credential Manager가 자동으로 인증을 처리할 수 있습니다:

```bash
cd "C:\Users\Admin\Downloads\Updated_streamlit_app_files"
git push -u origin main
```

브라우저가 자동으로 열리면 GitHub에 로그인하면 됩니다.

## 문제 해결

### "Authentication failed" 오류
- Personal Access Token을 정확히 입력했는지 확인
- 토큰이 만료되지 않았는지 확인
- `repo` 권한이 있는지 확인

### "Permission denied" 오류
- 저장소에 대한 쓰기 권한이 있는지 확인
- 저장소 소유자이거나 Collaborator 권한이 필요합니다

### "Repository not found" 오류
- 저장소 URL이 정확한지 확인: `https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits.git`
- 저장소가 Private인 경우 토큰에 적절한 권한이 있는지 확인

## 확인 방법

푸시가 성공하면 다음 링크에서 확인할 수 있습니다:
**https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits**

## 현재 커밋된 파일들

- ✅ `streamlit_app.py` - 모듈화된 메인 앱
- ✅ `requirements.txt` - 패키지 의존성
- ✅ `utils/` 폴더 전체 (8개 모듈)
- ✅ `.gitignore` - Git 제외 파일 목록








