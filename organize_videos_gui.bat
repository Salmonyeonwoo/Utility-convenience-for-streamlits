@echo off
chcp 65001 >nul
echo ========================================
echo 비디오 파일 구조화 도구 (GUI 버전)
echo ========================================
echo.

REM 현재 스크립트가 있는 디렉토리로 이동
cd /d "%~dp0"

REM Python 경로 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않거나 PATH에 등록되지 않았습니다.
    echo Python을 설치하거나 PATH에 추가한 후 다시 시도하세요.
    pause
    exit /b 1
)

REM GUI 스크립트 실행
python organize_videos_gui.py

if errorlevel 1 (
    echo.
    echo [오류] 스크립트 실행 중 오류가 발생했습니다.
    pause
)

























