@echo off
chcp 65001 >nul
echo ========================================
echo 비디오 파일 준비 폴더 생성
echo ========================================
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    pause
    exit /b 1
)

python create_video_structure.py

pause






