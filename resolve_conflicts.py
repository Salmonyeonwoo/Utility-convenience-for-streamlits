#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git merge conflict resolver - resolves conflicts by keeping the remote version
"""
import re

def resolve_conflicts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all conflicts
    pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [^\n]+\n'
    
    def replace_conflict(match):
        head_version = match.group(1)
        remote_version = match.group(2)
        # Keep remote version (after =======)
        return remote_version + '\n'
    
    # Replace all conflicts
    resolved_content = re.sub(pattern, replace_conflict, content, flags=re.DOTALL)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(resolved_content)
    
    # Count remaining conflicts
    remaining = len(re.findall(r'<<<<<<< HEAD', resolved_content))
    print(f"Resolved conflicts. Remaining: {remaining}")
    return remaining == 0

if __name__ == '__main__':
    resolve_conflicts('streamlit_app.py')












