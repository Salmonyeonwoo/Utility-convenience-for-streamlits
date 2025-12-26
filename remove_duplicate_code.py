# 중복 코드 제거 스크립트
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 회사 정보 탭 중복 코드 제거 (917줄부터 시작하는 중복 코드)
duplicate_start = None
for i, line in enumerate(lines):
    if '# 기존 회사 정보 탭 로직은' in line:
        duplicate_start = i
        break

if duplicate_start:
    # 다음 섹션 시작 전까지 찾기
    duplicate_end = None
    for i in range(duplicate_start + 1, len(lines)):
        if lines[i].strip() and not lines[i].startswith(' ') and not lines[i].startswith('\t'):
            if 'if feature_selection == L["sim_tab_chat_email"]:' in lines[i]:
                duplicate_end = i
                break
    
    if duplicate_end:
        # 중복 코드 제거
        new_lines = lines[:duplicate_start] + lines[duplicate_end:]
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"OK: Removed duplicate company info code (lines {duplicate_start+1}-{duplicate_end})")
    else:
        print("Warning: Could not find end of duplicate code")
else:
    print("No duplicate code found")








