"""간단한 conflict 마커 제거 스크립트"""
import re

file_path = "streamlit_app.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 모든 conflict 마커 제거 (HEAD 버전 유지)
# 패턴: <<<<<<< HEAD ... ======= ... >>>>>>>
pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n.*?\n>>>>>>> [^\n]+\n'
content = re.sub(pattern, r'\1\n', content, flags=re.DOTALL)

# 남은 단독 마커들 제거
content = re.sub(r'<<<<<<< HEAD\n', '', content)
content = re.sub(r'=======\n', '', content)
content = re.sub(r'>>>>>>> [^\n]+\n', '', content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ 충돌 마커 제거 완료!")






















