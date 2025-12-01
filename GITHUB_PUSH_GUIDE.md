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




