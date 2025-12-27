# syntax 오류 수정 스크립트
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 517줄부터 시작하는 잘못된 코드 블록 찾기
# 이 코드는 모듈로 이동되었으므로 제거해야 함
start_line = 517
end_line = None

# 잘못된 들여쓰기로 시작하는 코드 블록의 끝 찾기
# 다음 정상적인 코드 블록 시작 전까지
for i in range(start_line - 1, len(lines)):
    line = lines[i]
    # 들여쓰기가 없거나 적은 경우 (최상위 레벨 코드)
    stripped = line.lstrip()
    if stripped and not stripped.startswith('#'):
        indent = len(line) - len(stripped)
        # 들여쓰기가 0이거나 매우 작은 경우 (최상위 레벨)
        if indent <= 4 and ('if ' in stripped or 'def ' in stripped or 'class ' in stripped or 'import ' in stripped or 'from ' in stripped):
            # 하지만 이것이 잘못된 코드 블록의 일부가 아닌 경우
            if i > start_line + 50:  # 충분히 멀리 떨어진 경우
                end_line = i
                break

# 더 정확한 방법: 잘못된 들여쓰기 패턴 찾기
# 517줄부터 시작해서 들여쓰기가 36칸 이상인 코드 블록 찾기
if end_line is None:
    for i in range(start_line - 1, min(start_line + 200, len(lines))):
        line = lines[i]
        stripped = line.lstrip()
        if stripped and not stripped.startswith('#'):
            indent = len(line) - len(stripped)
            # 들여쓰기가 30칸 이상이면 잘못된 코드 블록의 일부
            if indent >= 30:
                continue
            # 들여쓰기가 0-4칸이고 의미있는 코드면 끝
            elif indent <= 4 and (stripped.startswith('if ') or stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('import ') or stripped.startswith('from ') or stripped.startswith('try:') or stripped.startswith('except') or stripped.startswith('with ') or stripped.startswith('for ') or stripped.startswith('while ')):
                if i > start_line:
                    end_line = i
                    break

# 여전히 찾지 못한 경우, 다음 주요 섹션 시작 전까지
if end_line is None:
    for i in range(start_line - 1, min(start_line + 300, len(lines))):
        line = lines[i]
        # 주요 섹션 시작 패턴 찾기
        if '# ========================================' in line or 'if feature_selection ==' in line or 'elif feature_selection ==' in line:
            if i > start_line + 10:  # 충분히 멀리 떨어진 경우
                end_line = i
                break

if end_line:
    # 잘못된 코드 블록 제거
    new_lines = lines[:start_line - 1] + lines[end_line:]
    with open('streamlit_app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"OK: Removed invalid code block (lines {start_line}-{end_line})")
else:
    print("Warning: Could not find end of invalid code block, trying alternative method")
    # 대안: 517줄부터 800줄 전까지 제거 (임시)
    if len(lines) > 800:
        new_lines = lines[:start_line - 1] + lines[800:]
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"OK: Removed code block (lines {start_line}-800)")










