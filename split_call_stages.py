# call_simulator.py를 IN_CALL과 CALL_ENDED 상태로 분리
import os

def extract_call_stage(input_file, output_file, start_marker, end_marker, header):
    """특정 마커 사이의 코드를 추출하여 새 파일로 저장"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
        if end_idx is None and start_idx is not None and end_marker in line:
            end_idx = i
            break
    
    if start_idx is None or end_idx is None:
        print(f"[ERROR] Could not find markers in {input_file}")
        return
    
    # 해당 범위의 코드 추출
    content = ''.join(lines[start_idx:end_idx])
    
    # 들여쓰기 조정 (elif 제거하고 함수 내부 코드로 변환)
    lines_content = content.split('\n')
    adjusted_lines = []
    for line in lines_content:
        if line.strip().startswith('elif'):
            # elif 제거하고 함수 내부 코드로 변환
            adjusted_lines.append('    ' + line.replace('elif st.session_state.call_sim_stage == "IN_CALL":', '').replace('elif st.session_state.call_sim_stage == "CALL_ENDED":', '').strip())
        elif line.strip() and not line.strip().startswith('#'):
            # 일반 코드는 들여쓰기 조정
            if line.startswith('        '):
                adjusted_lines.append('    ' + line[8:])  # 8칸 들여쓰기 제거하고 4칸 추가
            elif line.startswith('    '):
                adjusted_lines.append(line)
            else:
                adjusted_lines.append('    ' + line)
        else:
            adjusted_lines.append(line)
    
    full_content = header + '\n' + '\n'.join(adjusted_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print(f"[OK] {output_file} created ({end_idx - start_idx + 1} lines)")

# IN_CALL 상태 (99-1216줄)
in_call_header = '''# 전화 통화 중 상태 (IN_CALL)
import streamlit as st
from lang_pack import LANG
from datetime import datetime, timedelta
import numpy as np
from simulation_handler import *
from visualization import *
from audio_handler import *
from llm_client import get_api_key
from typing import List, Dict, Any
import uuid
import time
import os
from PIL import Image
import io

def render_call_in_call():
    """전화 통화 중 상태 렌더링"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
'''

extract_call_stage(
    'pages/call_simulator.py',
    'pages/call_in_call.py',
    'elif st.session_state.call_sim_stage == "IN_CALL":',
    'elif st.session_state.call_sim_stage == "CALL_ENDED":',
    in_call_header
)

# CALL_ENDED 상태 (1217-1369줄)
call_ended_header = '''# 전화 통화 종료 상태 (CALL_ENDED)
import streamlit as st
from lang_pack import LANG
from datetime import datetime, timedelta
from simulation_handler import *
import os

def render_call_ended():
    """전화 통화 종료 상태 렌더링"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
'''

extract_call_stage(
    'pages/call_simulator.py',
    'pages/call_ended.py',
    'elif st.session_state.call_sim_stage == "CALL_ENDED":',
    'def render_call_simulator():',  # 다음 함수 시작까지
    call_ended_header
)

print("[OK] All call stage files extracted!")




