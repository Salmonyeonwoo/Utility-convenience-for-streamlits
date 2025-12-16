@echo off
chcp 65001 >nul
echo ========================================
echo 테스트용 더미 비디오 파일 생성
echo ========================================
echo.
echo ⚠️  주의: 이것은 실제 비디오가 아닌 테스트용 파일입니다!
echo.
pause

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    pause
    exit /b 1
)

set /p TARGET_FOLDER="비디오 파일을 생성할 폴더 경로 (엔터: videos_to_organize): "

if "%TARGET_FOLDER%"=="" (
    python create_test_videos.py
) else (
    python create_test_videos.py "%TARGET_FOLDER%"
)

pause

























