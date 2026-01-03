# -*- coding: utf-8 -*-
import os
import shutil

streamlit_dir = r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\.streamlit"
config_file = os.path.join(streamlit_dir, "config.toml")

config_content = """[server]
port = 8501
address = "localhost"
headless = false
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
"""

print("=" * 50)
print("Streamlit config.toml 파일 생성")
print("=" * 50)
print()

# 디렉토리 생성
os.makedirs(streamlit_dir, exist_ok=True)
print(f"디렉토리 확인/생성: {streamlit_dir}")
print()

# 기존 파일 확인
if os.path.exists(config_file):
    print(f"기존 파일 발견: {config_file}")
    print("다음 중 선택하세요:")
    print("1. 기존 파일 백업 후 새로 생성")
    print("2. 기존 파일 덮어쓰기")
    print()
    
    # 백업 시도
    backup_file = config_file + ".backup"
    try:
        if os.path.exists(backup_file):
            os.remove(backup_file)
        shutil.copy2(config_file, backup_file)
        print(f"기존 파일 백업 완료: {backup_file}")
    except Exception as e:
        print(f"백업 실패 (무시): {e}")
    
    # 새 파일 생성 시도
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"새 파일 생성 성공!")
    except PermissionError:
        print("=" * 50)
        print("권한 오류 발생!")
        print("=" * 50)
        print()
        print("수동으로 처리하세요:")
        print()
        print(f"1. 파일 탐색기 열기")
        print(f"2. 다음 경로로 이동:")
        print(f"   {streamlit_dir}")
        print(f"3. config.toml 파일을 찾아서:")
        print(f"   - 마우스 오른쪽 클릭 -> 속성")
        print(f"   - '읽기 전용' 체크 해제 -> 확인")
        print(f"   - 파일 이름을 config.toml.old로 변경")
        print(f"4. 이 스크립트를 다시 실행하세요")
        exit(1)
else:
    print("기존 파일 없음. 새로 생성합니다.")
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"새 파일 생성 성공!")
    except Exception as e:
        print(f"파일 생성 실패: {e}")
        exit(1)

print()
print("=" * 50)
print("완료!")
print("=" * 50)
print(f"파일 경로: {config_file}")
print()
print("이제 Streamlit을 실행할 수 있습니다:")
print("  streamlit run streamlit_app.py")

