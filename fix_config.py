# -*- coding: utf-8 -*-
import os
import stat

# .streamlit 디렉토리 생성
streamlit_dir = r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\.streamlit"
os.makedirs(streamlit_dir, exist_ok=True)

# config.toml 파일 생성
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

try:
    print("config.toml 파일 생성 중...")
    
    # 기존 파일이 있으면 처리
    if os.path.exists(config_file):
        print("기존 파일 발견. 처리 중...")
        
        # 파일 권한 변경 시도
        try:
            os.chmod(config_file, stat.S_IWRITE | stat.S_IREAD)
            print("파일 권한 변경 성공")
        except:
            print("파일 권한 변경 실패 (무시)")
        
        # 삭제 시도
        try:
            os.remove(config_file)
            print("기존 파일 삭제 성공")
        except PermissionError:
            print("파일 삭제 실패. 이름 변경 시도...")
            # 이름 변경 시도
            try:
                backup_name = config_file + ".backup"
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                os.rename(config_file, backup_name)
                print("기존 파일을 .backup으로 이름 변경 성공")
            except Exception as e:
                print(f"이름 변경 실패: {e}")
                print("덮어쓰기 시도...")
                # 덮어쓰기 시도
                try:
                    with open(config_file, 'w', encoding='utf-8') as f:
                        f.write(config_content)
                    print("덮어쓰기 성공!")
                    print(f"파일 경로: {config_file}")
                    exit(0)
                except Exception as write_err:
                    print(f"덮어쓰기 실패: {write_err}")
                    print("\n수동 해결 방법:")
                    print(f"1. 파일 탐색기에서 다음 경로로 이동:")
                    print(f"   {streamlit_dir}")
                    print(f"2. config.toml 파일을 삭제하거나 이름 변경")
                    print(f"3. 관리자 권한으로 다시 실행하세요.")
                    exit(1)
    
    # 새 파일 생성
    print("새 config.toml 파일 생성 중...")
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("config.toml 파일이 성공적으로 생성되었습니다!")
    print(f"경로: {config_file}")
    
    # 파일 권한 설정
    try:
        os.chmod(config_file, stat.S_IWRITE | stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        print("파일 권한 설정 완료")
    except:
        print("파일 권한 설정 실패 (파일은 정상적으로 생성됨)")
    
    print("\n이제 Streamlit을 실행할 수 있습니다!")
    
except Exception as e:
    print(f"오류 발생: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\n수동 해결 방법:")
    print(f"1. 파일 탐색기에서 다음 경로로 이동:")
    print(f"   {streamlit_dir}")
    print("2. config.toml 파일이 있으면 삭제하거나 이름 변경")
    print("3. 새 config.toml 파일을 생성하여 위의 내용을 복사하세요.")
