"""5648번째 줄 이후의 잘못된 들여쓰기 코드 제거"""
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 5648번째 줄까지만 유지 (0-based index이므로 5647까지)
valid_lines = lines[:5648]

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(valid_lines)

print(f"✅ 파일 정리 완료: {len(valid_lines)}줄 유지")


























