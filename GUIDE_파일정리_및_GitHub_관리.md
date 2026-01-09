# 파일 정리 및 GitHub 관리 가이드라인

이 문서는 파일들을 타입별로 정리하고 GitHub에 효율적으로 커밋/푸시하는 방법을 설명합니다.

---

## 📁 1단계: 파일 타입별 폴더 생성 및 이동

### 방법 1: Windows 명령어 사용 (PowerShell 또는 CMD)

#### 1-1. 기본 폴더 구조 생성
```powershell
# 작업할 디렉토리로 이동
cd C:\Users\Admin\Downloads\Updated_streamlit_app_files\local_db

# 필요한 폴더들 한 번에 생성
mkdir docx, pptx, json, pdf, csv, txt, xlsx -Force
```

#### 1-2. 파일 타입별로 자동 이동 (PowerShell 스크립트)
```powershell
# docx 파일 이동
Get-ChildItem -Filter *.docx | Move-Item -Destination docx\

# pptx 파일 이동
Get-ChildItem -Filter *.pptx | Move-Item -Destination pptx\

# json 파일 이동
Get-ChildItem -Filter *.json | Move-Item -Destination json\

# pdf 파일 이동
Get-ChildItem -Filter *.pdf | Move-Item -Destination pdf\

# csv 파일 이동
Get-ChildItem -Filter *.csv | Move-Item -Destination csv\

# txt 파일 이동
Get-ChildItem -Filter *.txt | Move-Item -Destination txt\

# xlsx 파일 이동
Get-ChildItem -Filter *.xlsx | Move-Item -Destination xlsx\
```

#### 1-3. 한 번에 모든 파일 타입 정리 (통합 스크립트)
```powershell
# local_db 디렉토리에서 실행
$targetDir = "C:\Users\Admin\Downloads\Updated_streamlit_app_files\local_db"

# 폴더 생성
$folders = @("docx", "pptx", "json", "pdf", "csv", "txt", "xlsx")
foreach ($folder in $folders) {
    if (-not (Test-Path "$targetDir\$folder")) {
        New-Item -ItemType Directory -Path "$targetDir\$folder" -Force
    }
}

# 파일 타입별 이동
$fileTypes = @{
    "docx" = "*.docx"
    "pptx" = "*.pptx"
    "json" = "*.json"
    "pdf" = "*.pdf"
    "csv" = "*.csv"
    "txt" = "*.txt"
    "xlsx" = "*.xlsx"
}

foreach ($folder in $fileTypes.Keys) {
    Get-ChildItem -Path $targetDir -Filter $fileTypes[$folder] -File | 
        Move-Item -Destination "$targetDir\$folder\" -Force
    Write-Host "$folder 파일 이동 완료" -ForegroundColor Green
}
```

### 방법 2: Python 스크립트 사용 (더 세밀한 제어 가능)

`organize_files.py` 파일 생성:
```python
import os
import shutil
from pathlib import Path

def organize_files_by_type(source_dir, file_types):
    """
    파일 타입별로 폴더를 생성하고 파일을 이동합니다.
    
    Args:
        source_dir: 파일이 있는 소스 디렉토리
        file_types: {폴더명: [확장자 리스트]} 형식의 딕셔너리
    """
    source_path = Path(source_dir)
    
    # 폴더 생성
    for folder_name in file_types.keys():
        folder_path = source_path / folder_name
        folder_path.mkdir(exist_ok=True)
        print(f"✓ 폴더 생성/확인: {folder_name}")
    
    # 파일 이동
    moved_count = {folder: 0 for folder in file_types.keys()}
    
    for file_path in source_path.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower().lstrip('.')
            
            # 해당 확장자의 폴더 찾기
            for folder_name, extensions in file_types.items():
                if file_ext in extensions:
                    dest_path = source_path / folder_name / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    moved_count[folder_name] += 1
                    break
    
    # 결과 출력
    print("\n=== 이동 완료 ===")
    for folder, count in moved_count.items():
        if count > 0:
            print(f"{folder}: {count}개 파일 이동")
    
    total = sum(moved_count.values())
    print(f"\n총 {total}개 파일 이동 완료!")

# 사용 예시
if __name__ == "__main__":
    source_directory = r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\local_db"
    
    file_type_mapping = {
        "docx": ["docx"],
        "pptx": ["pptx"],
        "json": ["json"],
        "pdf": ["pdf"],
        "csv": ["csv"],
        "txt": ["txt"],
        "xlsx": ["xlsx", "xls"],
    }
    
    organize_files_by_type(source_directory, file_type_mapping)
```

실행:
```bash
python organize_files.py
```

---

## 🔍 2단계: 작업 전 확인

### 2-1. 현재 Git 상태 확인
```bash
cd C:\Users\Admin\Downloads\Updated_streamlit_app_files
git status
```

### 2-2. 변경사항 미리보기
```bash
# 스테이징되지 않은 변경사항 확인
git diff

# 파일 목록만 확인
git status --short
```

---

## 📦 3단계: Git에 파일 추가 (Staging)

### 3-1. 특정 폴더만 추가
```bash
# local_db의 특정 폴더만 추가
git add local_db/docx/
git add local_db/pptx/
git add local_db/json/
```

### 3-2. 특정 파일 타입만 추가
```bash
# 모든 json 파일 추가
git add local_db/**/*.json

# 모든 pdf 파일 추가
git add local_db/**/*.pdf
```

### 3-3. 모든 변경사항 추가 (주의!)
```bash
# 모든 변경사항 스테이징 (신중하게 사용)
git add .

# 또는 특정 디렉토리만
git add local_db/
```

### 3-4. 대화형 모드로 선택적 추가
```bash
# 파일 단위로 선택적으로 추가
git add -i

# 또는 패치 모드 (hunk 단위)
git add -p
```

---

## ✅ 4단계: 커밋 (Commit)

### 4-1. 기본 커밋
```bash
git commit -m "Add organized files: docx, pptx, json files sorted by type"
```

### 4-2. 상세한 커밋 메시지 (여러 줄)
```bash
git commit -m "Add organized local_db files" -m "
- Organized files by type into separate folders
- Added docx files to local_db/docx/
- Added pptx files to local_db/pptx/
- Added json files to local_db/json/
- Added pdf files to local_db/pdf/
- Added csv files to local_db/csv/
"
```

### 4-3. 변경된 파일 자동 포함 메시지
```bash
# 변경된 파일 목록을 포함한 커밋
git commit -m "Organize files by type" --verbose
```

---

## 🚀 5단계: GitHub에 푸시 (Push)

### 5-1. 기본 푸시
```bash
# main 브랜치에 푸시
git push origin main

# 또는 현재 브랜치에 푸시
git push
```

### 5-2. 원격 저장소 정보 확인
```bash
# 원격 저장소 목록 확인
git remote -v

# 원격 브랜치 확인
git branch -r
```

### 5-3. 푸시 전 최신 상태 확인 및 가져오기
```bash
# 원격 저장소의 변경사항 확인 (병합하지 않음)
git fetch origin

# 원격 변경사항과 비교
git log HEAD..origin/main

# 원격 변경사항 병합 후 푸시 (권장)
git pull origin main
git push origin main
```

---

## 🔄 통합 워크플로우 (한 번에 실행)

### PowerShell 스크립트: 전체 프로세스 자동화

`git_organize_and_push.ps1` 파일 생성:
```powershell
# 파일 정리 및 GitHub 푸시 스크립트

param(
    [string]$SourceDir = "C:\Users\Admin\Downloads\Updated_streamlit_app_files\local_db",
    [string]$CommitMessage = "Organize files by type and update repository",
    [string]$Branch = "main"
)

# 1. 파일 정리
Write-Host "=== 1단계: 파일 정리 시작 ===" -ForegroundColor Cyan

$folders = @("docx", "pptx", "json", "pdf", "csv", "txt", "xlsx")
foreach ($folder in $folders) {
    if (-not (Test-Path "$SourceDir\$folder")) {
        New-Item -ItemType Directory -Path "$SourceDir\$folder" -Force | Out-Null
        Write-Host "폴더 생성: $folder" -ForegroundColor Yellow
    }
}

$fileTypes = @{
    "docx" = "*.docx"
    "pptx" = "*.pptx"
    "json" = "*.json"
    "pdf" = "*.pdf"
    "csv" = "*.csv"
    "txt" = "*.txt"
    "xlsx" = "*.xlsx"
}

foreach ($folder in $fileTypes.Keys) {
    $files = Get-ChildItem -Path $SourceDir -Filter $fileTypes[$folder] -File -ErrorAction SilentlyContinue
    if ($files) {
        $files | Move-Item -Destination "$SourceDir\$folder\" -Force
        Write-Host "$folder`: $($files.Count)개 파일 이동" -ForegroundColor Green
    }
}

# 2. Git 상태 확인
Write-Host "`n=== 2단계: Git 상태 확인 ===" -ForegroundColor Cyan
$repoRoot = Split-Path -Parent $SourceDir
Set-Location $repoRoot
git status

# 3. 변경사항 추가
Write-Host "`n=== 3단계: 변경사항 스테이징 ===" -ForegroundColor Cyan
$response = Read-Host "모든 변경사항을 추가하시겠습니까? (Y/N)"
if ($response -eq "Y" -or $response -eq "y") {
    git add local_db/
    Write-Host "변경사항 스테이징 완료" -ForegroundColor Green
} else {
    Write-Host "스테이징 건너뜀" -ForegroundColor Yellow
    exit
}

# 4. 커밋
Write-Host "`n=== 4단계: 커밋 ===" -ForegroundColor Cyan
$customMessage = Read-Host "커밋 메시지를 입력하세요 (엔터 시 기본 메시지 사용)"
if ([string]::IsNullOrWhiteSpace($customMessage)) {
    $customMessage = $CommitMessage
}
git commit -m $customMessage
Write-Host "커밋 완료" -ForegroundColor Green

# 5. 푸시
Write-Host "`n=== 5단계: GitHub 푸시 ===" -ForegroundColor Cyan
$pushResponse = Read-Host "GitHub에 푸시하시겠습니까? (Y/N)"
if ($pushResponse -eq "Y" -or $pushResponse -eq "y") {
    git push origin $Branch
    Write-Host "푸시 완료!" -ForegroundColor Green
} else {
    Write-Host "푸시 건너뜀" -ForegroundColor Yellow
}

Write-Host "`n=== 모든 작업 완료 ===" -ForegroundColor Cyan
```

사용법:
```powershell
# 기본 실행
.\git_organize_and_push.ps1

# 커스텀 메시지와 함께
.\git_organize_and_push.ps1 -CommitMessage "Add new customer history files"
```

---

## ⚠️ 주의사항 및 Best Practices

### 1. 커밋 전 체크리스트
- [ ] `.gitignore`에 민감한 정보나 불필요한 파일이 제외되어 있는지 확인
- [ ] 커밋할 파일 목록 확인 (`git status`)
- [ ] 파일 크기가 너무 크지 않은지 확인 (GitHub는 100MB 이상 파일에 경고)
- [ ] 커밋 메시지가 명확하고 의미 있는지 확인

### 2. 커밋 메시지 작성 가이드
```
형식: [타입] 간단한 제목

상세 설명 (선택사항)

예시:
[Add] Organize local_db files by type
- Created separate folders for docx, pptx, json, pdf, csv files
- Moved 1,330 files to appropriate folders
```

### 3. 대용량 파일 처리
```bash
# Git LFS 사용 (100MB 이상 파일)
git lfs track "*.pdf"
git lfs track "*.pptx"
git add .gitattributes
```

### 4. 되돌리기 (실수했을 때)
```bash
# 스테이징 취소
git reset HEAD <file>

# 커밋 전 변경사항 되돌리기
git checkout -- <file>

# 마지막 커밋 취소 (변경사항 유지)
git reset --soft HEAD~1

# 마지막 커밋 완전히 취소
git reset --hard HEAD~1
```

### 5. 원격 푸시 취소 (주의!)
```bash
# 강제 푸시 (다른 사람과 협업 중이면 사용 금지!)
git push origin main --force
```

---

## 📋 빠른 참조 명령어

```bash
# 현재 상태 확인
git status

# 변경된 파일 확인
git diff

# 파일 추가
git add <file_or_folder>

# 커밋
git commit -m "메시지"

# 푸시
git push origin main

# 원격 변경사항 가져오기
git pull origin main

# 로그 확인
git log --oneline -10
```

---

## 🔧 문제 해결

### 문제: "remote: error: GH001: Large files detected"
**해결책:**
- 대용량 파일을 `.gitignore`에 추가하거나
- Git LFS를 사용하거나
- 파일을 삭제하고 다시 커밋

### 문제: "Updates were rejected because the remote contains work"
**해결책:**
```bash
git pull origin main --rebase
git push origin main
```

### 문제: 파일 이동 후 Git이 삭제/추가로 인식
**해결책:**
```bash
# Git이 파일 이동을 추적하도록 설정
git add -A
# 또는
git add --all
```

---

## 📚 추가 리소스

- [Git 공식 문서](https://git-scm.com/doc)
- [GitHub 가이드](https://guides.github.com/)
- [Git 커밋 메시지 가이드](https://chris.beams.io/posts/git-commit/)

---

**마지막 업데이트**: 2026-01-10
