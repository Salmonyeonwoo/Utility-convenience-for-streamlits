# streamlit_app.py 최종 정리 스크립트
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 세션 초기화 부분 찾기 (444줄부터 657줄 전까지)
session_start = 444
sidebar_start = 657

# 세션 초기화 코드를 함수 호출로 대체
new_lines = lines[:session_start-1]
new_lines.append('# 세션 상태 초기화\n')
new_lines.append('if SESSION_INIT_AVAILABLE:\n')
new_lines.append('    init_session_state()\n')
new_lines.append('\n')

# 사이드바 코드를 함수 호출로 대체
# 사이드바 끝 찾기 (다음 섹션 시작 전까지)
sidebar_end = None
for i in range(sidebar_start, len(lines)):
    if lines[i].strip() and not lines[i].startswith(' ') and not lines[i].startswith('\t'):
        if 'if feature_selection ==' in lines[i] or '# 메인 타이틀' in lines[i]:
            sidebar_end = i
            break

if sidebar_end is None:
    # 메인 타이틀 섹션 찾기
    for i in range(sidebar_start, len(lines)):
        if '# 메인 타이틀' in lines[i]:
            sidebar_end = i
            break

if sidebar_end:
    new_lines.append('# 사이드바 렌더링\n')
    new_lines.append('if SIDEBAR_AVAILABLE:\n')
    new_lines.append('    render_sidebar()\n')
    new_lines.append('\n')
    new_lines.extend(lines[sidebar_end:])
else:
    new_lines.extend(lines[sidebar_start-1:])

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("OK: Finalized streamlit_app.py")











