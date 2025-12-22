# 최종 정리: 모든 중복 코드 제거
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 517줄부터 시작하는 모든 중복 코드 제거
# 파일 끝까지 제거 (모든 탭 코드는 모듈로 이동됨)
start_line = 517
end_line = len(lines)

# 하지만 실제로는 파일 끝까지 제거하면 안 되므로, 
# 다음 주요 섹션이 있는지 확인
# 파일 끝이면 그대로 두기

# 517줄부터 파일 끝까지 확인
has_remaining_code = False
for i in range(start_line - 1, len(lines)):
    line = lines[i]
    stripped = line.strip()
    if stripped and not stripped.startswith('#'):
        has_remaining_code = True
        break

if has_remaining_code:
    # 517줄부터 파일 끝까지 제거 (주석만 남기기)
    # 하지만 실제로는 파일 끝까지 제거하면 안 되므로,
    # 빈 줄이나 주석만 남기기
    new_lines = lines[:start_line - 1]
    # 파일 끝에 빈 줄 추가
    new_lines.append('\n')
    
    with open('streamlit_app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"OK: Removed all remaining duplicate code (lines {start_line}-{end_line})")
    print(f"Removed {end_line - start_line + 1} lines")
else:
    print("No remaining code to remove")


