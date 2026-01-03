@echo off
echo ==========================================
echo Streamlit config.toml 파일 수정 스크립트
echo ==========================================
echo.
echo 관리자 권한이 필요할 수 있습니다.
echo.

cd /d "%~dp0"

REM 기존 파일 백업
if exist ".streamlit\config.toml" (
    echo 기존 파일을 config.toml.backup으로 백업 중...
    copy ".streamlit\config.toml" ".streamlit\config.toml.backup" >nul 2>&1
)

REM Python 스크립트 실행
python fix_config.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================
    echo Python 스크립트 실행 실패
    echo ==========================================
    echo.
    echo 수동으로 해결하는 방법:
    echo 1. 파일 탐색기에서 다음 경로로 이동:
    echo    %CD%\.streamlit
    echo.
    echo 2. config.toml 파일을 삭제하거나 이름 변경
    echo.
    echo 3. 다음 내용으로 새 config.toml 파일 생성:
    echo    [server]
    echo    port = 8501
    echo    address = "localhost"
    echo    headless = false
    echo    enableCORS = false
    echo    enableXsrfProtection = true
    echo.
    echo    [browser]
    echo    gatherUsageStats = false
    echo    serverAddress = "localhost"
    echo    serverPort = 8501
    echo.
    echo    [theme]
    echo    primaryColor = "#FF4B4B"
    echo    backgroundColor = "#FFFFFF"
    echo    secondaryBackgroundColor = "#F0F2F6"
    echo    textColor = "#262730"
    echo    font = "sans serif"
    echo.
    echo    [client]
    echo    showErrorDetails = true
    echo.
)

pause

