"""
Git 병합 충돌 마커 제거 스크립트
모듈화된 버전(HEAD)만 유지하고 원본 버전은 제거합니다.
"""
import re

file_path = "streamlit_app.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 충돌 마커 패턴
conflict_pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [^\n]+\n'

def replace_conflict(match):
    """충돌 구간에서 HEAD 부분만 유지"""
    head_content = match.group(1)
    return head_content + "\n"

# 충돌 마커 제거 (HEAD 부분만 유지)
content = re.sub(conflict_pattern, replace_conflict, content, flags=re.DOTALL)

# 남은 충돌 마커 제거 (단순 패턴)
content = re.sub(r'<<<<<<< HEAD\n', '', content)
content = re.sub(r'=======\n', '', content)
content = re.sub(r'>>>>>>> [^\n]+\n', '', content)

# 파일 저장
with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ 충돌 마커 제거 완료!")













