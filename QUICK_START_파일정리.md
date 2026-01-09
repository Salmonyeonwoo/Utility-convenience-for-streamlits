# 빠른 시작 가이드: 파일 정리 및 GitHub 푸시

## 🚀 가장 빠른 방법 (PowerShell 스크립트 사용)

```powershell
# 프로젝트 루트에서 실행
.\git_organize_and_push.ps1
```

## 📝 단계별 수동 실행

### 1. 파일 정리 (Python 스크립트)
```bash
python organize_files.py
```

### 2. Git 작업
```bash
# 상태 확인
git status

# 파일 추가
git add local_db/

# 커밋
git commit -m "Organize files by type"

# 푸시
git push origin main
```

## ⚡ PowerShell 한 줄 명령어

```powershell
# 파일 정리만
cd local_db; @("docx","pptx","json","pdf","csv","txt","xlsx") | ForEach-Object { mkdir $_ -Force; Get-ChildItem -Filter "*.$_" | Move-Item -Destination $_ }; cd ..

# Git 작업
git add local_db/; git commit -m "Organize files"; git push origin main
```

## 📋 체크리스트

- [ ] 파일 정리 완료
- [ ] `git status`로 변경사항 확인
- [ ] `.gitignore` 확인 (민감한 파일 제외)
- [ ] 커밋 메시지 작성
- [ ] 원격 저장소 동기화 확인
- [ ] 푸시 완료

## 🔗 상세 가이드

더 자세한 내용은 `GUIDE_파일정리_및_GitHub_관리.md` 파일을 참고하세요.
