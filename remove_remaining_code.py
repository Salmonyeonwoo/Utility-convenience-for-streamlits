# 남아있는 전화 시뮬레이터 및 기타 코드 제거 스크립트
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 517줄부터 시작하는 전화 시뮬레이터 로직 제거
# 다음 주요 섹션 시작 전까지
start_line = 517
end_line = None

# 다음 주요 섹션 찾기
for i in range(start_line - 1, len(lines)):
    line = lines[i]
    # RAG 탭 시작 찾기
    if 'elif feature_selection == L["rag_tab"]:' in line and 'if RAG_AVAILABLE' not in lines[i+1] if i+1 < len(lines) else True:
        # 이전 줄 확인
        if i > start_line + 10:
            end_line = i
            break

# 여전히 찾지 못한 경우, 다음 elif feature_selection 찾기
if end_line is None:
    for i in range(start_line - 1, len(lines)):
        line = lines[i]
        if 'elif feature_selection == L["rag_tab"]:' in line:
            if i > start_line + 10:
                end_line = i
                break

if end_line:
    # 잘못된 코드 블록 제거
    new_lines = lines[:start_line - 1] + lines[end_line:]
    with open('streamlit_app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"OK: Removed remaining phone simulator code (lines {start_line}-{end_line})")
    print(f"Removed {end_line - start_line + 1} lines")
else:
    print("Warning: Could not find end of remaining code")












