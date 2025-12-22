# -*- coding: utf-8 -*-
"""export 함수에서 os 문제 찾기"""

import re

with open('simulation_handler.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# export 함수들 찾기
exports = re.findall(r'def export_history_to_\w+', content)

for exp in exports:
    print(f"\n=== {exp} ===")
    # 함수 전체 찾기
    func_match = re.search(exp + r'\([^)]*\):.*?(?=def |\Z)', content, re.DOTALL)
    if func_match:
        func_content = func_match.group()
        lines = func_content.split('\n')
        
        # os import 확인
        has_import = False
        for i, line in enumerate(lines[:50]):
            if 'import os' in line or 'from os import' in line:
                has_import = True
                print(f"  os import 있음: 라인 {i+1}")
                break
        
        # os 사용 확인
        os_usage = []
        for i, line in enumerate(lines):
            if 'os.' in line or 'os.path' in line or 'os.path.join' in line:
                os_usage.append((i+1, line.rstrip()[:100]))
        
        if os_usage:
            print(f"  os 사용 위치:")
            for line_num, line_content in os_usage[:5]:
                print(f"    라인 {line_num}: {line_content}")
        
        if os_usage and not has_import:
            print(f"  ⚠️ 문제 발견: os를 사용하지만 import가 없음!")
            print(f"  함수 시작 부분:")
            for i, line in enumerate(lines[:20]):
                print(f"    {i+1}: {line.rstrip()[:80]}")



