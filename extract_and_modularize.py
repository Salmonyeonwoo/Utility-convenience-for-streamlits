#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
streamlit_app.py를 모듈화하는 스크립트
각 탭별로 별도 파일로 분리
"""

import os
import re

# 원본 파일 경로
SOURCE_FILE = "streamlit_app.py"
BACKUP_FILE = f"{SOURCE_FILE}.backup"

# 각 섹션의 시작 라인 번호 (grep 결과 기반)
SECTIONS = {
    "main_init": (1, 1467),  # 초기화 및 사이드바까지
    "company_info": (1468, 2521),  # 회사 정보 탭
    "chat_simulator": (2522, 6457),  # 채팅 시뮬레이터
    "phone_simulator": (6458, 8887),  # 전화 시뮬레이터
    "rag": (8888, 9159),  # RAG 탭
    "content": (9160, None),  # 콘텐츠 생성 탭 (끝까지)
}

def read_file_lines(filepath):
    """파일을 라인별로 읽기"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()

def extract_section(lines, start_line, end_line):
    """특정 라인 범위 추출 (1-based index)"""
    if end_line is None:
        return lines[start_line-1:]
    return lines[start_line-1:end_line]

def main():
    print("파일 모듈화 시작...")
    
    # 원본 파일 읽기
    if not os.path.exists(SOURCE_FILE):
        print(f"[ERROR] {SOURCE_FILE} 파일을 찾을 수 없습니다.")
        return
    
    print(f"[INFO] {SOURCE_FILE} 읽는 중...")
    all_lines = read_file_lines(SOURCE_FILE)
    total_lines = len(all_lines)
    print(f"[OK] 총 {total_lines}줄 읽음")
    
    # 백업 생성
    print(f"[INFO] 백업 생성 중...")
    with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
        f.writelines(all_lines)
    print(f"[OK] 백업 완료: {BACKUP_FILE}")
    
    # 각 섹션 추출
    for section_name, (start, end) in SECTIONS.items():
        print(f"\n[INFO] {section_name} 섹션 추출 중... (라인 {start}~{end or '끝'})")
        section_lines = extract_section(all_lines, start, end)
        
        # 파일로 저장
        output_file = f"pages/{section_name}.py"
        os.makedirs("pages", exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(section_lines)
        
        print(f"[OK] {output_file} 생성 완료 ({len(section_lines)}줄)")
    
    print("\n[OK] 모듈화 완료!")

if __name__ == "__main__":
    main()

