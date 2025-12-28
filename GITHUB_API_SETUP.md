# GitHub API 토큰 설정 가이드

## 1. GitHub Personal Access Token 생성하기

### 단계별 가이드:

1. **GitHub에 로그인**
   - https://github.com 에 로그인합니다.

2. **Settings로 이동**
   - 우측 상단 프로필 아이콘 클릭 → **Settings** 선택

3. **Developer settings 접근**
   - 좌측 메뉴 하단의 **Developer settings** 클릭
   - 또는 직접 접속: https://github.com/settings/developers

4. **Personal access tokens 생성**
   - **Personal access tokens** → **Tokens (classic)** 클릭
   - 또는 직접 접속: https://github.com/settings/tokens

5. **Generate new token**
   - **Generate new token** → **Generate new token (classic)** 클릭
   - GitHub 비밀번호 입력 요청 시 입력

6. **토큰 설정**
   - **Note**: 토큰 설명 입력 (예: "Streamlit App GitHub API")
   - **Expiration**: 만료 기간 선택 (30일, 60일, 90일, 또는 No expiration)
   - **Select scopes**: 필요한 권한 선택
     - ✅ **repo** (전체 저장소 접근) - 필수
     - 또는 더 제한적으로:
       - ✅ **public_repo** (공개 저장소만 접근)
       - ✅ **read:packages** (패키지 읽기)

7. **토큰 생성**
   - 하단의 **Generate token** 버튼 클릭
   - ⚠️ **중요**: 생성된 토큰을 즉시 복사하세요! (다시 볼 수 없습니다)
   - 토큰 형식: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

## 2. Streamlit 앱에 토큰 설정하기

### 방법 1: 로컬 개발 환경 (.streamlit/secrets.toml)

1. 프로젝트 루트의 `.streamlit/secrets.toml` 파일을 엽니다.

2. 다음 줄을 추가합니다:
```toml
GITHUB_API_KEY = "ghp_여기에_생성한_토큰_붙여넣기"
```

3. 파일 저장 후 Streamlit 앱을 재시작합니다.

### 방법 2: 환경 변수 설정 (Windows)

1. **시스템 환경 변수 설정**:
   - Windows 검색에서 "환경 변수" 검색
   - **시스템 환경 변수 편집** 선택
   - **환경 변수** 버튼 클릭
   - **시스템 변수** 섹션에서 **새로 만들기** 클릭
   - 변수 이름: `GITHUB_API_KEY`
   - 변수 값: 생성한 토큰
   - 확인 클릭

2. 또는 **PowerShell에서 임시 설정**:
```powershell
$env:GITHUB_API_KEY = "ghp_여기에_생성한_토큰_붙여넣기"
```

### 방법 3: Streamlit Cloud 배포 시

1. Streamlit Cloud 대시보드 접속: https://share.streamlit.io/

2. 앱 선택 → **Settings** → **Secrets** 탭

3. 다음 형식으로 추가:
```toml
GITHUB_API_KEY = "ghp_여기에_생성한_토큰_붙여넣기"
```

4. **Save** 클릭

## 3. 토큰 확인 및 테스트

토큰이 제대로 설정되었는지 확인하려면:

1. Streamlit 앱 실행
2. 채팅 시뮬레이터 페이지로 이동
3. 왼쪽 패널의 "파일에서 이력 불러오기" expander 열기
4. GitHub 파일이 표시되면 성공!

## 4. 보안 주의사항

⚠️ **중요한 보안 규칙**:

- ❌ **절대** GitHub 토큰을 코드에 직접 작성하지 마세요
- ❌ **절대** GitHub에 토큰을 커밋하지 마세요
- ✅ `.streamlit/secrets.toml`은 `.gitignore`에 포함되어 있어야 합니다
- ✅ 토큰이 유출되면 즉시 GitHub에서 토큰을 삭제하세요
- ✅ 토큰은 정기적으로 갱신하세요

## 5. 토큰이 없어도 작동하나요?

네, GitHub API는 토큰 없이도 작동합니다. 하지만:
- **Rate limit**: 시간당 60회 요청 제한 (토큰 있으면 5,000회)
- **인증된 접근**: 비공개 저장소 접근 불가

토큰을 설정하면 rate limit이 크게 증가하고, 더 많은 기능을 사용할 수 있습니다.

## 6. 문제 해결

### 토큰이 작동하지 않는 경우:

1. **토큰 형식 확인**: `ghp_`로 시작해야 합니다
2. **권한 확인**: `repo` 또는 `public_repo` 권한이 있는지 확인
3. **만료 확인**: 토큰이 만료되지 않았는지 확인
4. **앱 재시작**: secrets.toml 수정 후 앱을 재시작했는지 확인

### Rate limit 오류가 발생하는 경우:

- GitHub Personal Access Token을 설정하면 rate limit이 증가합니다
- 토큰 없이는 시간당 60회 제한이 있습니다





