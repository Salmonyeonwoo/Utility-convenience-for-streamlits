# 기존 회사 정보 탭 코드 제거 스크립트
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# if False: 블록 찾기 (882줄부터 시작)
start_idx = None
end_idx = None
indent_level = None

for i, line in enumerate(lines[881:], start=882):
    if 'if False:' in line and '# 기존 로직 비활성화' in line:
        start_idx = i - 1  # 0-based index
        indent_level = len(line) - len(line.lstrip())
        break

if start_idx is not None:
    # if False: 블록의 끝 찾기 (같은 들여쓰기 레벨의 다음 코드)
    for i, line in enumerate(lines[start_idx+1:], start=start_idx+1):
        # 빈 줄이 아니고, 들여쓰기가 같거나 작으면 끝
        stripped = line.lstrip()
        if stripped and not stripped.startswith('#'):
            line_indent = len(line) - len(stripped)
            if line_indent <= indent_level:
                end_idx = i
                break
    
    if end_idx is None:
        # 끝을 찾지 못한 경우, 다음 섹션 시작 전까지
        for i, line in enumerate(lines[start_idx+1:], start=start_idx+1):
            if '# ========================================' in line and '채팅/메일 시뮬레이터' in lines[i+1] if i+1 < len(lines) else False:
                end_idx = i
                break
    
    if end_idx:
        # 블록 제거
        new_lines = lines[:start_idx] + lines[end_idx:]
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"OK: Removed old company info code (lines {start_idx+1}-{end_idx})")
    else:
        print("Warning: Could not find end of if False block")
else:
    print("Warning: Could not find if False block")











