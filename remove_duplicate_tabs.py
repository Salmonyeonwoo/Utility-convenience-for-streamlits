# 중복된 탭 코드 제거 스크립트
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 517줄부터 시작하는 중복된 RAG 탭 코드 제거
# 다음 주요 섹션 시작 전까지
start_line = 517
end_line = None

# 다음 elif feature_selection 찾기
for i in range(start_line - 1, len(lines)):
    line = lines[i]
    if 'elif feature_selection == L["content_tab"]:' in line:
        if i > start_line + 10:
            end_line = i
            break

# 여전히 찾지 못한 경우, 파일 끝까지
if end_line is None:
    end_line = len(lines)

if end_line:
    # 중복 코드 블록 제거
    new_lines = lines[:start_line - 1] + lines[end_line:]
    with open('streamlit_app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"OK: Removed duplicate tab code (lines {start_line}-{end_line})")
    print(f"Removed {end_line - start_line + 1} lines")
else:
    print("Warning: Could not find end of duplicate code")








