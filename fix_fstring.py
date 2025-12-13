# f-string 오류 수정 스크립트
import re

with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 빈 f-string 찾기 및 수정
for i in range(len(lines)):
    line = lines[i]
    # 빈 f-string 패턴 찾기: f""" 다음에 바로 """ 또는 빈 줄
    if 'f"""' in line:
        # 다음 몇 줄 확인
        j = i + 1
        found_content = False
        while j < len(lines) and j < i + 10:
            if '"""' in lines[j]:
                # f"""와 """ 사이에 내용이 없으면 일반 문자열로 변경
                if not found_content:
                    lines[i] = line.replace('f"""', '"""')
                break
            if lines[j].strip() and not lines[j].strip().startswith('#'):
                found_content = True
            j += 1

# 파일 저장
with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('f-string 오류 수정 완료')

