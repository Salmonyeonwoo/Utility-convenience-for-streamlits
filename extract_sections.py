# 파일 분리 스크립트
import os

def extract_section(input_file, output_file, start_line, end_line, header_imports):
    """특정 라인 범위를 추출하여 새 파일로 저장"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 해당 범위의 코드 추출
    content = ''.join(lines[start_line-1:end_line])
    
    # 헤더와 import 추가
    full_content = header_imports + '\n\n' + content
    
    # 들여쓰기 조정 (원본 코드가 elif로 시작하므로 함수 내부로 이동)
    lines_content = full_content.split('\n')
    adjusted_lines = []
    for line in lines_content:
        if line.strip().startswith('elif feature_selection'):
            # elif 제거하고 함수 내부 코드로 변환
            adjusted_lines.append('    ' + line.replace('elif feature_selection == L["sim_tab_chat_email"]:', '').strip())
        elif line.strip() and not line.strip().startswith('#'):
            # 일반 코드는 들여쓰기 추가
            if not line.startswith(' '):
                adjusted_lines.append('    ' + line)
            else:
                adjusted_lines.append(line)
        else:
            adjusted_lines.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(adjusted_lines))
    
    print(f"[OK] {output_file} created ({end_line - start_line + 1} lines)")

# 채팅/이메일 시뮬레이터 (1161-4562줄)
chat_header = '''# 채팅/이메일 시뮬레이터
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

def render_chat_simulator():
    """채팅/이메일 시뮬레이터 렌더링"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
'''

extract_section(
    'streamlit_app.py',
    'pages/chat_simulator.py',
    1161,
    4562,
    chat_header
)

# 전화 시뮬레이터 (4578-6197줄)  
call_header = '''# 전화 시뮬레이터
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

def render_call_simulator():
    """전화 시뮬레이터 렌더링 (전화 수신, 문의 입력 포함)"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
'''

extract_section(
    'streamlit_app.py',
    'pages/call_simulator.py',
    4578,
    6197,
    call_header
)

# 콘텐츠 생성기 (6212-6871줄)
content_header = '''# 콘텐츠 생성기
import streamlit as st
from lang_pack import LANG
from llm_client import get_api_key, run_llm
from openai import OpenAI
import json
import uuid

def render_content_generator():
    """콘텐츠 생성기 렌더링"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
'''

extract_section(
    'streamlit_app.py',
    'pages/content_generator.py',
    6212,
    6871,
    content_header
)

print("[OK] All files extracted!")

