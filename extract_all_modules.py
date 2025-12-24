# 모든 탭을 모듈로 추출하는 스크립트
import re

with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 각 탭의 시작과 끝 라인 찾기
tabs = {
    'phone': {'start': None, 'end': None, 'name': '_phone_simulator'},
    'rag': {'start': None, 'end': None, 'name': '_rag'},
    'content': {'start': None, 'end': None, 'name': '_content'},
    'lstm': {'start': None, 'end': None, 'name': '_lstm'},
    'voice': {'start': None, 'end': None, 'name': '_voice_rec'},
}

# 각 탭의 시작 라인 찾기
for i, line in enumerate(lines):
    if 'elif feature_selection == L["sim_tab_phone"]:' in line and 'st.header' not in line:
        # 타이틀 부분은 건너뛰고 실제 코드 시작 찾기
        for j in range(i+1, len(lines)):
            if 'elif feature_selection == L["sim_tab_phone"]:' in lines[j] and 'st.header' in lines[j]:
                tabs['phone']['start'] = j
                break
    elif 'elif feature_selection == L["rag_tab"]:' in line and 'st.header' in line:
        tabs['rag']['start'] = i
    elif 'elif feature_selection == L["content_tab"]:' in line and 'st.header' in line:
        tabs['content']['start'] = i
    elif 'elif feature_selection == L["lstm_tab"]:' in line and 'st.header' in line:
        tabs['lstm']['start'] = i
    elif 'elif feature_selection == L["voice_rec_header"]:' in line and 'st.header' in line:
        tabs['voice']['start'] = i

# 각 탭의 끝 라인 찾기 (다음 탭 시작 전까지)
tab_order = ['phone', 'rag', 'content', 'lstm', 'voice']
for idx, tab_key in enumerate(tab_order):
    if tabs[tab_key]['start'] is not None:
        # 다음 탭의 시작을 끝으로 설정
        next_tab_start = None
        for next_idx in range(idx + 1, len(tab_order)):
            if tabs[tab_order[next_idx]]['start'] is not None:
                next_tab_start = tabs[tab_order[next_idx]]['start']
                break
        
        if next_tab_start:
            tabs[tab_key]['end'] = next_tab_start
        else:
            # 마지막 탭인 경우, 파일 끝까지
            tabs[tab_key]['end'] = len(lines)

# 전화 시뮬레이터 탭 추출
if tabs['phone']['start'] and tabs['phone']['end']:
    phone_code = ''.join(lines[tabs['phone']['start']:tabs['phone']['end']])
    
    module_content = f"""# ========================================
# _pages/_phone_simulator.py
# 전화 시뮬레이터 탭 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from datetime import datetime, timedelta
import time
import os
import base64
import tempfile
from typing import List, Dict, Any
from simulation_handler import *
from video_handler import *
from audio_handler import *
from llm_client import get_api_key, run_llm, init_openai_audio_client
from config import *
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai
import requests

def render_phone_simulator():
    \"\"\"전화 시뮬레이터 탭 렌더링 함수\"\"\"
{phone_code}
"""
    
    with open('_pages/_phone_simulator.py', 'w', encoding='utf-8') as f:
        f.write(module_content)
    print(f"OK: Phone simulator module created: _pages/_phone_simulator.py ({tabs['phone']['end']-tabs['phone']['start']} lines)")

# RAG 탭 추출
if tabs['rag']['start'] and tabs['rag']['end']:
    rag_code = ''.join(lines[tabs['rag']['start']:tabs['rag']['end']])
    
    module_content = f"""# ========================================
# _pages/_rag.py
# RAG 탭 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from datetime import datetime
import os
from rag_handler import *
from simulation_handler import load_simulation_histories_local
from llm_client import get_api_key, run_llm
from config import DATA_DIR, RAG_INDEX_DIR

def render_rag():
    \"\"\"RAG 탭 렌더링 함수\"\"\"
{rag_code}
"""
    
    with open('_pages/_rag.py', 'w', encoding='utf-8') as f:
        f.write(module_content)
    print(f"OK: RAG module created: _pages/_rag.py ({tabs['rag']['end']-tabs['rag']['start']} lines)")

# 콘텐츠 탭 추출
if tabs['content']['start'] and tabs['content']['end']:
    content_code = ''.join(lines[tabs['content']['start']:tabs['content']['end']])
    
    module_content = f"""# ========================================
# _pages/_content.py
# 콘텐츠 탭 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
import json
import re
import numpy as np
from llm_client import run_llm
try:
    import plotly.graph_objects as go
    import plotly.express as px
    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False

def render_content():
    \"\"\"콘텐츠 탭 렌더링 함수\"\"\"
{content_code}
"""
    
    with open('_pages/_content.py', 'w', encoding='utf-8') as f:
        f.write(module_content)
    print(f"OK: Content module created: _pages/_content.py ({tabs['content']['end']-tabs['content']['start']} lines)")

print("All modules extracted successfully!")






