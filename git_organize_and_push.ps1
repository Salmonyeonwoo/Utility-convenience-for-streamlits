# 파일 정리 및 GitHub 푸시 통합 스크립트
# 사용법: .\git_organize_and_push.ps1

param(
    [string]$SourceDir = "C:\Users\Admin\Downloads\Updated_streamlit_app_files\local_db",
    [string]$RepoRoot = "C:\Users\Admin\Downloads\Updated_streamlit_app_files",
    [string]$CommitMessage = "",
    [string]$Branch = "main",
    [switch]$SkipOrganize = $false,
    [switch]$SkipCommit = $false,
    [switch]$AutoConfirm = $false
)

function Write-Step {
    param([string]$Message, [string]$Color = "Cyan")
    Write-Host "`n========================================" -ForegroundColor $Color
    Write-Host $Message -ForegroundColor $Color
    Write-Host "========================================`n" -ForegroundColor $Color
}

function Confirm-Action {
    param([string]$Message)
    if ($AutoConfirm) { return $true }
    $response = Read-Host "$Message (Y/N)"
    return ($response -eq "Y" -or $response -eq "y")
}

# 1. 파일 정리
if (-not $SkipOrganize) {
    Write-Step "1단계: 파일 정리 시작"
    
    if (-not (Test-Path $SourceDir)) {
        Write-Host "❌ 오류: 소스 디렉토리를 찾을 수 없습니다: $SourceDir" -ForegroundColor Red
        exit 1
    }
    
    # 폴더 생성
    $folders = @("docx", "pptx", "json", "pdf", "csv", "txt", "xlsx")
    foreach ($folder in $folders) {
        $folderPath = Join-Path $SourceDir $folder
        if (-not (Test-Path $folderPath)) {
            New-Item -ItemType Directory -Path $folderPath -Force | Out-Null
            Write-Host "✓ 폴더 생성: $folder" -ForegroundColor Yellow
        }
    }
    
    # 파일 이동
    $fileTypes = @{
        "docx" = "*.docx"
        "pptx" = "*.pptx"
        "json" = "*.json"
        "pdf" = "*.pdf"
        "csv" = "*.csv"
        "txt" = "*.txt"
        "xlsx" = "*.xlsx"
    }
    
    $totalMoved = 0
    foreach ($folder in $fileTypes.Keys) {
        $files = Get-ChildItem -Path $SourceDir -Filter $fileTypes[$folder] -File -ErrorAction SilentlyContinue
        if ($files) {
            $files | Move-Item -Destination (Join-Path $SourceDir $folder) -Force
            Write-Host "✓ $folder`: $($files.Count)개 파일 이동" -ForegroundColor Green
            $totalMoved += $files.Count
        }
    }
    
    if ($totalMoved -eq 0) {
        Write-Host "이동할 파일이 없습니다." -ForegroundColor Yellow
    } else {
        Write-Host "`n✅ 총 $totalMoved 개 파일 이동 완료!" -ForegroundColor Green
    }
} else {
    Write-Host "`n⏭️  파일 정리 단계 건너뜀" -ForegroundColor Yellow
}

# 2. Git 상태 확인
Write-Step "2단계: Git 상태 확인"

if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
    Write-Host "❌ 오류: Git 저장소를 찾을 수 없습니다: $RepoRoot" -ForegroundColor Red
    exit 1
}

Set-Location $RepoRoot
$gitStatus = git status --short

if ([string]::IsNullOrWhiteSpace($gitStatus)) {
    Write-Host "변경사항이 없습니다." -ForegroundColor Yellow
    Write-Host "작업을 종료합니다." -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "변경된 파일:" -ForegroundColor Cyan
    git status --short
}

# 3. 변경사항 추가
Write-Step "3단계: 변경사항 스테이징"

if (-not $SkipCommit) {
    if (Confirm-Action "모든 변경사항을 스테이징하시겠습니까?") {
        git add local_db/
        $stagedFiles = git diff --cached --name-only
        if ($stagedFiles) {
            Write-Host "✅ 스테이징 완료" -ForegroundColor Green
            Write-Host "스테이징된 파일 수: $($stagedFiles.Count)" -ForegroundColor Cyan
        } else {
            Write-Host "⚠️  스테이징할 파일이 없습니다." -ForegroundColor Yellow
        }
    } else {
        Write-Host "스테이징이 취소되었습니다." -ForegroundColor Yellow
        exit 0
    }
} else {
    Write-Host "⏭️  스테이징 단계 건너뜀" -ForegroundColor Yellow
}

# 4. 커밋
if (-not $SkipCommit) {
    Write-Step "4단계: 커밋"
    
    $commitMsg = $CommitMessage
    if ([string]::IsNullOrWhiteSpace($commitMsg)) {
        $commitMsg = Read-Host "커밋 메시지를 입력하세요 (엔터 시 기본 메시지)"
        if ([string]::IsNullOrWhiteSpace($commitMsg)) {
            $commitMsg = "Organize files by type and update repository"
        }
    }
    
    Write-Host "커밋 메시지: $commitMsg" -ForegroundColor Cyan
    
    if (Confirm-Action "커밋하시겠습니까?") {
        git commit -m $commitMsg
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ 커밋 완료" -ForegroundColor Green
        } else {
            Write-Host "❌ 커밋 실패" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "커밋이 취소되었습니다." -ForegroundColor Yellow
        exit 0
    }
} else {
    Write-Host "`n⏭️  커밋 단계 건너뜀" -ForegroundColor Yellow
}

# 5. 푸시
Write-Step "5단계: GitHub 푸시"

# 원격 저장소 확인
$remoteUrl = git remote get-url origin 2>$null
if ($remoteUrl) {
    Write-Host "원격 저장소: $remoteUrl" -ForegroundColor Cyan
} else {
    Write-Host "❌ 오류: 원격 저장소를 찾을 수 없습니다." -ForegroundColor Red
    exit 1
}

# 원격 변경사항 확인
Write-Host "원격 변경사항 확인 중..." -ForegroundColor Cyan
git fetch origin 2>&1 | Out-Null

$localCommit = git rev-parse HEAD
$remoteCommit = git rev-parse "origin/$Branch" 2>$null

if ($remoteCommit -and $localCommit -ne $remoteCommit) {
    Write-Host "⚠️  원격 저장소에 새로운 커밋이 있습니다." -ForegroundColor Yellow
    if (Confirm-Action "원격 변경사항을 가져와서 병합하시겠습니까?") {
        git pull origin $Branch --rebase
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ 병합 중 충돌이 발생했습니다. 수동으로 해결해주세요." -ForegroundColor Red
            exit 1
        }
    }
}

# 푸시 실행
if (Confirm-Action "GitHub에 푸시하시겠습니까?") {
    Write-Host "푸시 중..." -ForegroundColor Cyan
    git push origin $Branch
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 푸시 완료!" -ForegroundColor Green
    } else {
        Write-Host "❌ 푸시 실패" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "푸시가 취소되었습니다." -ForegroundColor Yellow
}

# 최종 상태 확인
Write-Step "최종 상태 확인"
git status

Write-Host "`n" + ("=" * 60) -ForegroundColor Green
Write-Host "✅ 모든 작업 완료!" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
