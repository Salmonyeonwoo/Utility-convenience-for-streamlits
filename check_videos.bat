@echo off
chcp 65001 >nul
echo ========================================
echo 비디오 파일 검색 진단 도구
echo ========================================
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    pause
    exit /b 1
)

set /p FOLDER_PATH="확인할 폴더 경로를 입력하세요: "

if "%FOLDER_PATH%"=="" (
    echo [오류] 폴더 경로를 입력해야 합니다.
    pause
    exit /b 1
)

python check_videos.py "%FOLDER_PATH%"

pause










