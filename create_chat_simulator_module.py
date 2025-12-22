# 임시 스크립트: pages/chat_simulator.py를 _pages/_chat_simulator.py로 변환
import os

# _pages 폴더 생성
os.makedirs("_pages", exist_ok=True)

# pages/chat_simulator.py 읽기
with open("pages/chat_simulator.py", "r", encoding="utf-8") as f:
    content = f.read()

# 함수로 감싸기
header = """# ========================================
# _pages/_chat_simulator.py
# 채팅/이메일 시뮬레이터 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import *
from visualization import *
from llm_client import get_api_key
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time
import re
import os
import tempfile
import json
import base64
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader


def render_chat_simulator():
    \"\"\"채팅/이메일 시뮬레이터 렌더링 함수\"\"\"
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # pages/chat_simulator.py의 내용 (if feature_selection == L["sim_tab_chat_email"]: 제거)
"""

# if feature_selection == L["sim_tab_chat_email"]: 제거
content = content.replace('if feature_selection == L["sim_tab_chat_email"]:', '')

# 들여쓰기 조정 (4칸 제거하고 함수 내부로 들여쓰기 추가)
lines = content.split('\n')
adjusted_lines = []
for line in lines:
    if line.strip():  # 빈 줄이 아니면
        if line.startswith('    '):  # 4칸 들여쓰기가 있으면 제거하고 함수 내부로 들여쓰기
            adjusted_lines.append('    ' + line[4:])  # 함수 내부로 4칸 들여쓰기 추가
        else:
            adjusted_lines.append('    ' + line)  # 들여쓰기가 없으면 함수 내부로 4칸 들여쓰기 추가
    else:
        adjusted_lines.append(line)

content = '\n'.join(adjusted_lines)

# 파일 저장
with open("_pages/_chat_simulator.py", "w", encoding="utf-8") as f:
    f.write(header + content)

print("Success: _pages/_chat_simulator.py created")

