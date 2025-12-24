# 고아가 된 코드 블록 제거 스크립트
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 517줄부터 시작하는 elif 블록 찾기
# 이 코드는 모듈로 이동되었으므로 제거해야 함
start_line = 517
end_line = None

# elif로 시작하는 블록의 끝 찾기
# 다음 주요 섹션 시작 전까지
for i in range(start_line - 1, len(lines)):
    line = lines[i]
    stripped = line.lstrip()
    
    # 주요 섹션 시작 패턴 찾기
    if '# ========================================' in line:
        if i > start_line + 10:  # 충분히 멀리 떨어진 경우
            end_line = i
            break
    # elif가 아닌 최상위 레벨 코드 찾기
    elif stripped and not stripped.startswith('#'):
        indent = len(line) - len(stripped)
        if indent == 0:
            # 최상위 레벨 코드
            if (stripped.startswith('if ') or stripped.startswith('def ') or 
                stripped.startswith('class ') or stripped.startswith('import ') or 
                stripped.startswith('from ') or stripped.startswith('try:') or
                'if feature_selection' in stripped):
                if i > start_line + 10:
                    end_line = i
                    break

# 여전히 찾지 못한 경우, 다음 feature_selection 체크 전까지
if end_line is None:
    for i in range(start_line - 1, len(lines)):
        line = lines[i]
        if 'if feature_selection ==' in line or 'elif feature_selection ==' in line:
            if i > start_line + 10:
                end_line = i
                break

if end_line:
    # 잘못된 코드 블록 제거
    new_lines = lines[:start_line - 1] + lines[end_line:]
    with open('streamlit_app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"OK: Removed orphaned code block (lines {start_line}-{end_line})")
    print(f"Removed {end_line - start_line + 1} lines")
else:
    print("Warning: Could not find end of orphaned code block")






