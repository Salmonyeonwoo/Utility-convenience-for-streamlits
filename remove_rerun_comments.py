# -*- coding: utf-8 -*-
"""
주석 처리된 st.rerun() 제거 스크립트
"""
import os
import re

def remove_rerun_comments(file_path):
    """파일에서 주석 처리된 st.rerun() 제거"""
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 주석 처리된 st.rerun() 패턴 제거
    # 패턴 1: # st.rerun()  # 주석...
    content = re.sub(r'^\s*#\s*st\.rerun\(\)\s*#.*$', '', content, flags=re.MULTILINE)
    # 패턴 2: # st.rerun()
    content = re.sub(r'^\s*#\s*st\.rerun\(\)\s*$', '', content, flags=re.MULTILINE)
    # 패턴 3: # ⭐ ... st.rerun() ...
    content = re.sub(r'^\s*#\s*.*st\.rerun\(\)\s*$', '', content, flags=re.MULTILINE)
    
    # 빈 줄이 3개 이상 연속으로 있으면 2개로 줄이기
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

# 주요 파일 목록 (백업 파일 제외)
main_files = [
    '_pages/_chat_agent_turn.py',
    '_pages/_chat_initial_query.py',
    '_pages/_content.py',
    '_pages/_content_generator.py',
    '_pages/_chat_role_selection.py',
    '_pages/chat_modules/guideline_draft_customer.py',
    '_pages/chat_modules/customer_turn.py',
    '_pages/chat_modules/customer_closing_response.py',
    '_pages/chat_modules/closing_confirmation.py',
    '_pages/chat_modules/agent_turn.py',
]

base_dir = os.path.dirname(os.path.abspath(__file__))
removed_count = 0

for file_path in main_files:
    full_path = os.path.join(base_dir, file_path)
    if remove_rerun_comments(full_path):
        print(f"✅ {file_path}")
        removed_count += 1
    else:
        print(f"⏭️  {file_path} (변경 없음)")

print(f"\n완료: {removed_count}개 파일에서 주석 처리된 rerun 제거됨")

