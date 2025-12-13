@echo off
chcp 65001 >nul
echo ========================================
echo 비디오 파일 구조화 스크립트 (대화형 모드)
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

REM 비디오 폴더 경로 입력 받기
set /p VIDEO_DIR="비디오 파일이 있는 폴더 경로를 입력하세요: "

if "%VIDEO_DIR%"=="" (
    echo [오류] 폴더 경로를 입력해야 합니다.
    pause
    exit /b 1
)

REM 폴더 존재 확인
if not exist "%VIDEO_DIR%" (
    echo [오류] 지정한 폴더가 존재하지 않습니다: %VIDEO_DIR%
    pause
    exit /b 1
)

echo.
echo 비디오 파일 구조화를 시작합니다 (대화형 모드)...
echo 소스 폴더: %VIDEO_DIR%
echo 파일명에서 정보를 추출할 수 없는 경우 수동으로 입력하라는 메시지가 표시됩니다.
echo.

REM 스크립트 실행 (대화형 모드)
python organize_videos.py "%VIDEO_DIR%" -i

echo.
echo ========================================
echo 작업이 완료되었습니다!
echo ========================================
pause















