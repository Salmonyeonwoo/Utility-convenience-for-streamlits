"""
로컬 파일과 GitHub 파일 비교 스크립트
"""
import os
import requests
import hashlib
from pathlib import Path

GITHUB_REPO = "Salmonyeonwoo/Utility-convenience-for-streamlits"
GITHUB_BRANCH = "main"
LOCAL_PAGES_DIR = r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\_pages"
LOCAL_UTILS_DIR = r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\utils"

def get_file_hash(filepath):
    """파일의 해시값 계산"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def get_github_file_content(filepath):
    """GitHub에서 파일 내용 가져오기"""
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{filepath}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        return None
    except Exception:
        return None

def get_github_file_hash(filepath):
    """GitHub 파일의 해시값 계산"""
    content = get_github_file_content(filepath)
    if content:
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    return None

def compare_directories():
    """로컬과 GitHub의 파일 비교"""
    
    # 주요 파일 목록
    pages_files = [
        "_pages/_chat_simulator.py",
        "_pages/_chat_agent_turn.py",
        "_pages/_chat_customer_turn.py",
        "_pages/_chat_closing.py",
        "_pages/_chat_history.py",
        "_pages/_chat_initial_query.py",
        "_pages/_chat_messages.py",
        "_pages/_chat_role_selection.py",
        "_pages/_chat_transfer.py",
        "_pages/_chat_customer_message.py",
        "_pages/_chat_styles.py",
        "_pages/_content.py",
        "_pages/_content_generator.py",
    ]
    
    utils_files = [
        "utils/history_handler.py",
        "utils/customer_analysis.py",
        "utils/customer_verification.py",
        "utils/audio_handler.py",
    ]
    
    all_files = pages_files + utils_files
    
    same_count = 0
    different_count = 0
    missing_count = 0
    
    print("=" * 60)
    print("로컬과 GitHub 파일 비교 시작")
    print("=" * 60)
    
    for filepath in all_files:
        local_path = filepath.replace("_pages/", LOCAL_PAGES_DIR + "\\").replace("utils/", LOCAL_UTILS_DIR + "\\")
        
        if not os.path.exists(local_path):
            print(f"[없음] {filepath} - 로컬에 파일이 없습니다.")
            missing_count += 1
            continue
        
        local_hash = get_file_hash(local_path)
        github_hash = get_github_file_hash(filepath)
        
        if github_hash is None:
            print(f"[확인불가] {filepath} - GitHub에서 가져올 수 없습니다.")
            continue
        
        if local_hash == github_hash:
            same_count += 1
            print(f"[동일] {filepath}")
        else:
            different_count += 1
            print(f"[다름] {filepath}")
    
    print("=" * 60)
    print(f"비교 완료: 동일={same_count}, 다름={different_count}, 없음={missing_count}")
    print("=" * 60)
    
    if different_count == 0 and missing_count == 0:
        return "예"
    else:
        return "아니오"

if __name__ == "__main__":
    result = compare_directories()
    print(f"\n최종 결과: {result}")


