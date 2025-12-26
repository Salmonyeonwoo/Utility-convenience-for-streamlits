"""
GitHub 저장소에서 채팅 시뮬레이터 및 맞춤형 콘텐츠 생성 기능 복원 스크립트
"""
import os
import requests
import zipfile
import shutil
from pathlib import Path

GITHUB_REPO = "Salmonyeonwoo/Utility-convenience-for-streamlits"
GITHUB_BRANCH = "main"
TARGET_DIR = r"C:\Users\Admin\Downloads\Updated_streamlit_app_files"
TEMP_DIR = r"C:\Users\Admin\Downloads\temp_github_restore"

def download_github_zip():
    """GitHub 저장소를 ZIP으로 다운로드"""
    url = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"
    
    print(f"다운로드 중: {url}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        zip_path = os.path.join(TEMP_DIR, "repo.zip")
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r진행률: {percent:.1f}%", end='', flush=True)
        
        print(f"\n다운로드 완료: {zip_path}")
        return zip_path
    except Exception as e:
        print(f"\n다운로드 실패: {e}")
        raise

def extract_and_copy_files(zip_path):
    """ZIP 압축 해제 및 필요한 파일 복사"""
    extract_path = os.path.join(TEMP_DIR, "extracted")
    
    # 압축 해제
    print("\n압축 해제 중...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except Exception as e:
        print(f"압축 해제 실패: {e}")
        raise
    
    repo_dir = os.path.join(extract_path, f"Utility-convenience-for-streamlits-{GITHUB_BRANCH}")
    
    if not os.path.exists(repo_dir):
        raise FileNotFoundError(f"저장소 디렉토리를 찾을 수 없습니다: {repo_dir}")
    
    # 복사할 파일 목록
    files_to_copy = [
        ("_pages/_chat_simulator.py", "_pages"),
        ("_pages/_chat_agent_turn.py", "_pages"),
        ("_pages/_chat_customer_turn.py", "_pages"),
        ("_pages/_chat_closing.py", "_pages"),
        ("_pages/_chat_history.py", "_pages"),
        ("_pages/_chat_initial_query.py", "_pages"),
        ("_pages/_chat_messages.py", "_pages"),
        ("_pages/_chat_role_selection.py", "_pages"),
        ("_pages/_chat_transfer.py", "_pages"),
        ("_pages/_chat_customer_message.py", "_pages"),
        ("_pages/_chat_styles.py", "_pages"),
        ("_pages/_content.py", "_pages"),
        ("_pages/_content_generator.py", "_pages"),
        ("simulation_handler.py", ""),
        ("simulation_perspective_logic.py", ""),
    ]
    
    # 파일 복사
    print("\n파일 복사 중...")
    copied_count = 0
    for file_path, target_subdir in files_to_copy:
        source = os.path.join(repo_dir, file_path)
        if os.path.exists(source):
            target_dir = os.path.join(TARGET_DIR, target_subdir) if target_subdir else TARGET_DIR
            os.makedirs(target_dir, exist_ok=True)
            target = os.path.join(target_dir, os.path.basename(file_path))
            
            # 백업 (기존 파일이 있으면)
            if os.path.exists(target):
                backup = target + ".backup"
                shutil.copy2(target, backup)
                print(f"  백업: {os.path.basename(target)} → {os.path.basename(backup)}")
            
            shutil.copy2(source, target)
            print(f"[OK] {file_path}")
            copied_count += 1
        else:
            print(f"[SKIP] 파일 없음: {file_path}")
    
    # chat_modules 폴더 전체 복사
    chat_modules_source = os.path.join(repo_dir, "_pages", "chat_modules")
    chat_modules_target = os.path.join(TARGET_DIR, "_pages", "chat_modules")
    if os.path.exists(chat_modules_source):
        if os.path.exists(chat_modules_target):
            backup_target = chat_modules_target + "_backup"
            if os.path.exists(backup_target):
                shutil.rmtree(backup_target)
            shutil.copytree(chat_modules_target, backup_target)
            print(f"  백업: chat_modules/ → chat_modules_backup/")
            shutil.rmtree(chat_modules_target)
        shutil.copytree(chat_modules_source, chat_modules_target)
        print("[OK] _pages/chat_modules/ (전체 폴더)")
        copied_count += 1
    
    # utils 폴더의 파일들 복사
    utils_source = os.path.join(repo_dir, "utils")
    utils_target = os.path.join(TARGET_DIR, "utils")
    if os.path.exists(utils_source):
        os.makedirs(utils_target, exist_ok=True)
        utils_files = [f for f in os.listdir(utils_source) if f.endswith('.py')]
        for file in utils_files:
            source_file = os.path.join(utils_source, file)
            target_file = os.path.join(utils_target, file)
            
            # 백업
            if os.path.exists(target_file):
                backup_file = target_file + ".backup"
                shutil.copy2(target_file, backup_file)
            
            shutil.copy2(source_file, target_file)
            print(f"[OK] utils/{file}")
            copied_count += 1
    
    print(f"\n총 {copied_count}개 파일/폴더 복사 완료")

def cleanup():
    """임시 파일 정리"""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"\n임시 파일 삭제 완료: {TEMP_DIR}")
        except Exception as e:
            print(f"\n임시 파일 삭제 실패 (수동 삭제 필요): {e}")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("GitHub 저장소에서 기능 복원 시작")
    print("=" * 60)
    print(f"저장소: {GITHUB_REPO}")
    print(f"브랜치: {GITHUB_BRANCH}")
    print(f"대상 디렉토리: {TARGET_DIR}")
    print("=" * 60)
    
    try:
        # 타겟 디렉토리 확인
        if not os.path.exists(TARGET_DIR):
            print(f"\n[ERROR] 대상 디렉토리가 존재하지 않습니다: {TARGET_DIR}")
            return
        
        # 다운로드 및 복사
        zip_path = download_github_zip()
        extract_and_copy_files(zip_path)
        cleanup()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 복원 완료!")
        print("=" * 60)
        print("\n다음 단계:")
        print("1. streamlit run streamlit_app.py 로 실행 테스트")
        print("2. Import 오류가 있으면 수동으로 수정")
        print("3. 백업 파일(*.backup)은 안전 확인 후 삭제")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n수동 복원을 시도하세요:")
        print("1. https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits 접속")
        print("2. Code → Download ZIP 클릭")
        print("3. 압축 해제 후 필요한 파일을 수동으로 복사")

if __name__ == "__main__":
    main()

