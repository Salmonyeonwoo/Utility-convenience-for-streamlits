# 모든 syntax 오류 확인 및 수정 스크립트
import ast
import sys

with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

# Python AST로 syntax 오류 확인
try:
    ast.parse(content)
    print("OK: No syntax errors found!")
except SyntaxError as e:
    print(f"Syntax error found at line {e.lineno}: {e.msg}")
    print(f"Text: {e.text}")
    
    # 오류가 있는 라인 주변 확인
    error_line = e.lineno - 1
    start = max(0, error_line - 5)
    end = min(len(lines), error_line + 5)
    
    print(f"\nContext around line {e.lineno}:")
    for i in range(start, end):
        marker = ">>> " if i == error_line else "    "
        print(f"{marker}{i+1:4d}: {lines[i]}")
    
    sys.exit(1)











