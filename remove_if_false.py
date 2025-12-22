# if False 블록 제거 스크립트
import re

def remove_if_false_blocks(input_file, output_file):
    """if False 블록과 그 안의 코드를 제거"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # if False 블록 찾기 및 제거
    # 패턴: "if False:" 다음에 오는 모든 들여쓰기된 코드 블록 제거
    lines = content.split('\n')
    result_lines = []
    skip_block = False
    indent_level = 0
    
    for i, line in enumerate(lines):
        # if False 블록 시작 감지
        if re.match(r'^\s*if False\s*:', line):
            skip_block = True
            indent_level = len(line) - len(line.lstrip())
            continue
        
        # 블록 내부 코드 스킵
        if skip_block:
            current_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 1
            # 같은 레벨의 다른 코드가 나오면 스킵 종료
            if line.strip() and current_indent <= indent_level and not line.strip().startswith('#'):
                skip_block = False
                result_lines.append(line)
            # 빈 줄이나 주석은 유지하지 않음
            continue
        
        result_lines.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_lines))
    
    print(f"[OK] Removed if False blocks from {input_file}")

remove_if_false_blocks('streamlit_app.py', 'streamlit_app.py')




