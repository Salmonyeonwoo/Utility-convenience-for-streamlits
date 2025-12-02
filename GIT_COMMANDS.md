# GitHub 커밋 및 푸시 명령어 순서

## 기본 명령어 순서

### 1단계: 프로젝트 디렉토리로 이동
```bash
cd "c:\Users\Admin\Downloads\Updated_streamlit_app_files"
```

### 2단계: 현재 상태 확인
```bash
git status
```

### 3단계: 변경사항 스테이징 (모든 파일 추가)
```bash
git add .
```

또는 특정 파일만 추가하려면:
```bash
git add 파일명
```

### 4단계: 커밋 생성
```bash
git commit -m "커밋 메시지"
```

예시:
```bash
git commit -m "코드 업데이트 및 유틸리티 스크립트 추가"
```

### 5단계: GitHub에 푸시
```bash
git push -u origin main
```

---

## 전체 명령어 한 번에 실행 (복사해서 사용)

```bash
cd "c:\Users\Admin\Downloads\Updated_streamlit_app_files"
git status
git add .
git commit -m "커밋 메시지 입력"
git push -u origin main
```

---

## 유용한 추가 명령어

### 원격 저장소 확인
```bash
git remote -v
```

### 커밋 히스토리 확인
```bash
git log --oneline -5
```

### 변경사항 취소 (스테이징 전)
```bash
git restore 파일명
```

### 스테이징 취소 (커밋 전)
```bash
git restore --staged 파일명
```

---

## 인증 방법

### Personal Access Token 사용 시
1. GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token (classic)" 클릭
3. Note 입력, `repo` 권한 체크
4. 토큰 생성 후 복사
5. `git push` 실행 시:
   - Username: GitHub 사용자명
   - Password: **Personal Access Token** (일반 비밀번호 아님!)

---

## 저장소 정보
- 원격 저장소: https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits.git
- 브랜치: main





